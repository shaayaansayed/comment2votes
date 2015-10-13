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

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a character-level language model')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data_dir','data/test','data directory. Should contain the file input.txt with input data')
-- model params
cmd:option('-rnn_size', 128, 'size of LSTM internal state')
cmd:option('-num_layers', 2, 'number of layers in the LSTM')
cmd:option('-model', 'lstm', 'lstm,gru or rnn')
-- optimization
cmd:option('-learning_rate',2e-3,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',10,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-dropout',0,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-batch_size',50,'number of sequences to train on in parallel')
cmd:option('-max_epochs',10,'number of full passes through the training data')
cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:option('-train_frac',0.95,'fraction of data that goes into train set')
cmd:option('-val_frac',0.05,'fraction of data that goes into validation set')
            -- test_frac will be computed as (1 - train_frac - val_frac)
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
-- bookkeeping
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-print_every',1,'how many steps/minibatches between printing out the loss')
cmd:option('-eval_val_every',1000,'every how many iterations should we evaluate on validation data?')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-savefile','lstm','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
-- GPU/CPU
cmd:option('-gpuid',-1,'which gpu to use. -1 = use CPU')
cmd:text()

opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

local test_frac = math.max(0, 1 - (opt.train_frac + opt.val_frac))
local split_sizes = {opt.train_frac, opt.val_frac, test_frac} 
local loader = Comment2VoteSGDLoader.create(opt.data_dir, opt.batch_size, split_sizes)
vocab_size = loader.vocab_size
assert(loader:largest_comment_size() <= 500)

if opt.gpuid >=0 then
  local ok, cunn = pcall(require, 'cunn')
  assert(ok, 'package cunn not found!')
  local ok, cutorch = pcall(require, 'cutorch')
  cutorch.setDevice(opt.gpuid+1) 
end

encode, decode = {}, {}
encode.rnn = LSTM.create(vocab_size, vocab_size, opt.rnn_size, opt.num_layers, 0)
decode.rnn = LSTM.create(1, 1, opt.rnn_size, opt.num_layers, 1)
decode.criterion = nn.MSECriterion()

if opt.gpuid >=0 then
    for k,v in pairs(encode) do v:cuda() end
    for k,v in pairs(decode) do v:cuda() end
end

params, grad_params = model_utils.combine_all_parameters(encode.rnn, decode.rnn)
params:uniform(-.08, .08)

print('cloning encoder and decoder...')
dclone = {}
for name,proto in pairs(decode) do
	dclone[name] = model_utils.clone_many_times(proto, 1, not proto.parameters)
end

eclone = {}
local max_enc_len = loader:largest_comment_size()
for name, proto in pairs(encode) do
  eclone[name] = model_utils.clone_many_times(proto, max_enc_len, not proto.parameters)
end

local init_state = {}
for i=1, opt.num_layers do
	local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
	if opt.gpuid >=0 then h_init = h_init:cuda() end
	table.insert(init_state, h_init:clone())
	table.insert(init_state, h_init:clone())
end

print('number of parameters in the model: ' .. params:nElement())

local t_vec = torch.DoubleTensor(opt.batch_size, 1)
if opt.gpuid >= 0 then t_vec = t_vec:cuda() end
t_vec:fill(-1)

function eval_split(split_index, max_batches)
    print('evaluating loss over validation set...')
    local n = loader.split_sizes[split_index]
    if max_batches ~= nil then n = math.min(max_batches, n) end

    loader:reset_batch_pointer(split_index) -- move batch iteration pointer for this split to front
    local loss = 0
	local en_rnn_state = {[0] = init_state}
    
    for i = 1,n do -- iterate over batches in the split
        -- fetch a batch
        local x, y = loader:next_batch(split_index)

        if opt.gpuid >= 0 then
	  		x = x:float():cuda()
	  		y = y:float():cuda()
		end

        -- forward pass
		for t=1,in_length do
			eclone.rnn[t]:training()
			local lst = eclone.rnn[t]:forward{x[{{}, t}], unpack(en_rnn_state[t-1])}
			en_rnn_state[t] = {}
			for k=1, #init_state do table.insert(en_rnn_state[t], lst[k]) end
		end

		local dec_rnn_state = {[0] = en_rnn_state[#en_rnn_state]}
		dclone.rnn[1]:training()
		local lst = dclone.rnn[1]:forward{t_vec, unpack(dec_rnn_state[0])}
		dec_rnn_state[1] = {}
		for k=1, #init_state do table.insert(dec_rnn_state[1], lst[k]) end
		local loss = loss + dclone.criterion[1]:forward(lst[#lst], y)
		print(i .. '/' .. n .. '...' )
        -- carry over lstm state
        en_rnn_state[0] = en_rnn_state[#en_rnn_state]
    end

    loss = loss / n
    print('total average validation loss: ' .. loss)
    return loss
end

init_global_state = clone_list(init_state)
function feval(x)
	-- update paramaters
	if x ~= params then
	    params:copy(x)
	end
	-- reset gradients
	grad_params:zero()

	-- get minibatch
	--local x, y = loader:next_batch()
	local x, y = loader:next_batch(1)
	in_length = x:select(1,1):nElement()
	assert(in_length <= max_enc_len)

	if opt.gpuid >= 0 then
	  x = x:float():cuda()
	  y = y:float():cuda()
	end

	---------- FORWARD PASS ---------------------
	-- clone encoding LSTM
	local en_rnn_state = {[0] = init_global_state}
	for t=1,in_length do
		eclone.rnn[t]:training()
		local lst = eclone.rnn[t]:forward{x[{{}, t}], unpack(en_rnn_state[t-1])}
		en_rnn_state[t] = {}
		for k=1, #init_state do table.insert(en_rnn_state[t], lst[k]) end
	end

	local dec_rnn_state = {[0] = en_rnn_state[#en_rnn_state]}
	dclone.rnn[1]:training()
	local lst = dclone.rnn[1]:forward{t_vec, unpack(dec_rnn_state[0])}
	dec_rnn_state[1] = {}
	for k=1, #init_state do table.insert(dec_rnn_state[1], lst[k]) end
	local loss = dclone.criterion[1]:forward(lst[#lst], y)

	-- -- -- ------- BACKWARD PASS --------------------
	local ddec_rnn_state = {[1] = clone_list(init_state)}
	doutput = dclone.criterion[1]:backward(lst[#lst], y)
	table.insert(ddec_rnn_state[1], doutput)
	local dh = dclone.rnn[1]:backward({t_vec, unpack(dec_rnn_state[0])}, ddec_rnn_state[1])
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

	init_state_global = en_rnn_state[#en_rnn_state]
	grad_params:clamp(-opt.grad_clip, opt.grad_clip)

	return loss, grad_params
end

train_losses = {}
val_losses = {}
local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
local iterations = opt.max_epochs * loader.ntrain
local iterations_per_epoch = loader.ntrain
local loss0 = nil

for i = 1, iterations do
    local epoch = i / loader.ntrain

    local timer = torch.Timer()
    local _, loss = optim.rmsprop(feval, params, optim_state)
    local time = timer:time().real

    local train_loss = loss[1] -- the loss is inside a list, pop it
    train_losses[i] = train_loss

	if opt.accurate_gpu_timing == 1 and opt.gpuid >= 0 then
        cutorch.synchronize()
    end

    -- exponential learning rate decay
    if i % loader.ntrain == 0 and opt.learning_rate_decay < 1 then
        if epoch >= opt.learning_rate_decay_after then
            local decay_factor = opt.learning_rate_decay
            optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
            print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
        end
    end

    if i % opt.eval_val_every == 0 or i == iterations then
        -- evaluate loss on validation data
        local val_loss = eval_split(2) -- 2 = validation
        val_losses[i] = val_loss

        local savefile = string.format('%s/lm_%s_epoch%.2f_%.4f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss)
        print('saving checkpoint to ' .. savefile)
        local checkpoint = {}
        checkpoint.protos = protos
        checkpoint.opt = opt
        checkpoint.train_losses = train_losses
        checkpoint.val_loss = val_loss
        checkpoint.val_losses = val_losses
        checkpoint.i = i
        checkpoint.epoch = epoch
        checkpoint.vocab = loader.vocab_mapping
        torch.save(savefile, checkpoint)
    end

    if i % opt.print_every == 0 then
        print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.2fs", i, iterations, epoch, train_loss, grad_params:norm() / params:norm(), time))
    end
   
    if i % 10 == 0 then collectgarbage() end
end
