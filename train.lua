require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

require 'util.OneHot'
require 'util.misc'

local model_utils = require 'util.model_utils'
local Comment2VoteSGDLoader = require 'util.Comment2VoteSGDLoader'
local LSTM = require 'LSTM'

rnn_size = 32
num_layers = 1
grad_clip = 5
learning_rate = 2e-3
learning_rate_decay = .97
decay_rate = .95
learning_rate_decay_after = 10000
batch_size = 128

local split_sizes = {1, 0, 0} 
--data_dir = 'data/nba2'
data_dir = 'data/test'
local loader = Comment2VoteSGDLoader.create(data_dir, batch_size, split_sizes)
vocab_size = loader.vocab_size

gpuid = 1
if gpuid >=0 then
  local ok, cunn = pcall(require, 'cunn')
  assert(ok, 'package cunn not found!')
  local ok, cutorch = pcall(require, 'cutorch')
  cutorch.setDevice(gpuid+1) 
end

encode, decode = {}, {}
encode.rnn = LSTM.create(vocab_size, vocab_size, rnn_size, num_layers, 0)
decode.rnn = LSTM.create(1, 1, rnn_size, num_layers, 1)
decode.criterion = nn.MSECriterion()

if gpuid >=0 then
    for k,v in pairs(encode) do v:cuda() end
    for k,v in pairs(decode) do v:cuda() end
end

params, grad_params = model_utils.combine_all_parameters(encode.rnn, decode.rnn)
params:uniform(-.08, .08)


dclone = {}
for name,proto in pairs(decode) do
	dclone[name] = model_utils.clone_many_times(proto, 1, not proto.parameters)
end

eclone = {}
local max_enc_len = 128
for name, proto in pairs(encode) do
  eclone[name] = model_utils.clone_many_times(proto, max_enc_len, not proto.parameters)
end

local init_state_enc = {}
local init_state_dec = {}
for i=1, num_layers do
	local h_init = torch.zeros(batch_size, rnn_size)
	if gpuid >=0 then h_init = h_init:cuda() end
	table.insert(init_state_enc, h_init:clone())
	table.insert(init_state_enc, h_init:clone())
	table.insert(init_state_dec, h_init:clone())
	table.insert(init_state_dec, h_init:clone())
end

print('number of parameters in the model: ' .. params:nElement())

local t_vec = torch.DoubleTensor(batch_size, 1)
if gpuid >= 0 then t_vec = t_vec:cuda() end
t_vec:fill(-1)

init_global_state_enc = clone_list(init_state_enc)
init_global_state_dec = clone_list(init_state_dec)

function feval(x)
	-- update paramaters
	if x ~= params then
	    params:copy(x)
	end
	-- reset gradients
	grad_params:zero()

	-- get minibatch
	--local x, y = loader:next_batch()
	local x, y = loader:nextbatch()
	in_length = x:select(1,1):nElement()
	assert(in_length <= max_enc_len)

	if gpuid >= 0 then
	  x = x:float():cuda()
	  y = y:float():cuda()
	end

	---------- FORWARD PASS ---------------------
	-- clone encoding LSTM
	local en_rnn_state = {[0] = init_global_state_enc}
	for t=1,in_length do
		eclone.rnn[t]:training()
		local lst = eclone.rnn[t]:forward{x[{{}, t}], unpack(en_rnn_state[t-1])}
		en_rnn_state[t] = {}
		for k=1, #init_state_enc do table.insert(en_rnn_state[t], lst[k]) end
	end

	--local dec_rnn_state = {[0] = en_rnn_state[#en_rnn_state]}
	local dec_rnn_state = {[0] = init_global_state_dec}
	dclone.rnn[1]:training()
	--local lst = dclone.rnn[1]:forward{t_vec, unpack(dec_rnn_state[0])}
	local hv_out = en_rnn_state[#en_rnn_state]
	local lst = dclone.rnn[1]:forward{hv_out[#hv_out], unpack(dec_rnn_state[0])}
	dec_rnn_state[1] = {}
	for k=1, #init_state_dec do table.insert(dec_rnn_state[1], lst[k]) end
	local loss = dclone.criterion[1]:forward(lst[#lst], y)

	-- -- -- ------- BACKWARD PASS --------------------
	local ddec_rnn_state = {[1] = clone_list(init_state_dec)}
	doutput = dclone.criterion[1]:backward(lst[#lst], y)
	table.insert(ddec_rnn_state[1], doutput)
	local dh = dclone.rnn[1]:backward({hv_out, unpack(dec_rnn_state[1])}, 
	  ddec_rnn_state[1])
	ddec_rnn_state[0] = {}
	for k=1, #init_state_dec do table.insert(ddec_rnn_state[0], dh[k+1]) end

	local den_rnn_state = {[in_length] = clone_list(ddec_rnn_state[0])}
	for t=in_length, 1, -1 do
		local dh = eclone.rnn[t]:backward({x[{{}, t}], unpack(en_rnn_state[t-1])}, den_rnn_state[t])
		den_rnn_state[t-1] = {}
		for k=1, #init_state_dec do
			table.insert(den_rnn_state[t-1], dh[k+1])
		end
	end

	init_state_global_enc = en_rnn_state[#en_rnn_state]
	init_state_global_dec = dec_rnn_state[#dec_rnn_state]

	grad_params:clamp(-grad_clip, grad_clip)
	return loss, grad_params
end

train_losses = {}
val_losses = {}
local optim_state = {learningRate = learning_rate, alpha = decay_rate}
local iterations = 500 * loader.ntrain
local iterations_per_epoch = loader.ntrain
local loss0 = nil

state = {
   learningRate = 1e-3,
   momentum = 0.5
}

for i = 1, iterations do
    local epoch = i / loader.ntrain

    local timer = torch.Timer()
    -- local _, loss = optim.sgd(feval, params, state)
    local _, loss = optim.rmsprop(feval, params, optim_state)
    local time = timer:time().real

    local train_loss = loss[1] -- the loss is inside a list, pop it
    train_losses[i] = train_loss

    -- exponential learning rate decay
    if i % loader.ntrain == 0 and learning_rate_decay < 1 then
        if epoch >= learning_rate_decay_after then
            local decay_factor = learning_rate_decay
            optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
            print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
        end
	loader.batch_ix = 0
    end

    if i % 20 == 0 then
        print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.2fs", i, iterations, epoch, train_loss, grad_params:norm() / params:norm(), time))
    end
   
    if i % 10 == 0 then collectgarbage() end

    -- handle early stopping if things are going really bad
    if loss[1] ~= loss[1] then
        print('loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
        break -- halt
    end
    if loss0 == nil then loss0 = loss[1] end
    -- if loss[1] > loss0 * 3 then
    --     print('loss is exploding, aborting.')
    --     break -- halt
    -- end
end
