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

rnn_size = 5
num_layers = 1
grad_clip = 5
learning_rate = 2e-3
learning_rate_decay = .97
decay_rate = .95
learning_rate_decay_after = 10
batch_size = 128

local split_sizes = {1, 0, 0} 
data_dir = 'data/nba2'
local loader = Comment2VoteSGDLoader.create(data_dir, batch_size, split_sizes)
vocab_size = loader.vocab_size

encode, decode = {}, {}
encode.rnn = LSTM.create(vocab_size, vocab_size, rnn_size, num_layers, 0)
decode.rnn = LSTM.create(1, 1, rnn_size, num_layers, 1)
decode.criterion = nn.MSECriterion()

params, grad_params = model_utils.combine_all_parameters(encode.rnn, decode.rnn)
params:uniform(-.08, .08)

dclone = {}
for name,proto in pairs(decode) do
	dclone[name] = model_utils.clone_many_times(proto, 1, not proto.parameters)
end

local init_state = {}
for i=1, num_layers do
	local h_init = torch.zeros(batch_size, rnn_size)
	table.insert(init_state, h_init:clone())
	table.insert(init_state, h_init:clone())
end

print('number of parameters in the model: ' .. params:nElement())

init_global_state = clone_list(init_state)
function feval(x)
	-- update paramaters
	if x ~= params then
	    params:copy(x)
	end
	-- reset gradients
	grad_params:zero()

	-- get minibatch
	local x, y = loader:nextbatch()
	in_length = x:select(1,1):nElement()

	---------- FORWARD PASS ---------------------
	-- clone encoding LSTM
	eclone = {}
	for name, proto in pairs(encode) do
		eclone[name] = model_utils.clone_many_times(proto, in_length, not proto.parameters)
	end
	local en_rnn_state = {[0] = init_global_state}
	for t=1,in_length do
		eclone.rnn[t]:training()
		local lst = eclone.rnn[t]:forward{x[{{}, t}], unpack(en_rnn_state[t-1])}
		en_rnn_state[t] = {}
		for k=1, #init_state do table.insert(en_rnn_state[t], lst[k]) end
	end

	local dec_rnn_state = {[0] = en_rnn_state[#en_rnn_state]}
	dclone.rnn[1]:training()
	local lst = dclone.rnn[1]:forward{torch.DoubleTensor(batch_size, 1):fill(-1), unpack(dec_rnn_state[0])}
	dec_rnn_state[1] = {}
	for k=1, #init_state do table.insert(dec_rnn_state[1], lst[k]) end
	local loss = dclone.criterion[1]:forward(lst[#lst], y)

	-- -- -- ------- BACKWARD PASS --------------------
	local ddec_rnn_state = {[1] = clone_list(init_state)}
	doutput = dclone.criterion[1]:backward(lst[#lst], y)
	table.insert(ddec_rnn_state[1], doutput)
	local dh = dclone.rnn[1]:backward({torch.DoubleTensor({-1}):view(1,1), unpack(dec_rnn_state[0])}, ddec_rnn_state[1])
	ddec_rnn_state[0] = {}
	for k=1, #init_state do table.insert(ddec_rnn_state[0], dh[k+1]) end

	local den_rnn_state = {[in_length] = clone_list(ddec_rnn_state[0])}
	for t=in_length, 1, -1 do
		local dh = eclone.rnn[t]:backward({x[{{}, t}], unpack(en_rnn_state[t-1])}, den_rnn_state[t])
		den_rnn_state[t-1] = {}
		for k=1, #init_state do
			table.insert(den_rnn_state[t-1], dh[k+1])
		end
	end

	grad_params:clamp(-grad_clip, grad_clip)
	return loss, grad_params
end

train_losses = {}
val_losses = {}
local optim_state = {learningRate = learning_rate, alpha = decay_rate}
local iterations = 1 * loader.ntrain
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
    end

    if i % 1 == 0 then
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