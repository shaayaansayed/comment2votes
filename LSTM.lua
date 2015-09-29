local LSTM = {}
function LSTM.create(input_size, output_size, rnn_size, n, is_dec)
	local inputs = {}
	local outputs = {}

	table.insert(inputs, nn.Identity()())
	for L=1, n do
		table.insert(inputs, nn.Identity()())
		table.insert(inputs, nn.Identity()())
	end

	local x, input_size_L
	for L=1, n do
		if L == 1 then
			if is_dec == 1 then
				x = inputs[1]
			else
				x = OneHot(input_size)(inputs[1])
			end
			input_size_L = input_size
		else 
			x = outputs[(L-1)*2]
			input_size_L = rnn_size
		end
		prev_c = inputs[L*2]
		prev_h = inputs[L*2 + 1]

		local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x)
		local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h)
		local all_input_sums = nn.CAddTable()({i2h, h2h})

		local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
		local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)

		local in_gate = nn.Sigmoid()(n1)
		local forget_gate = nn.Sigmoid()(n2)
		local out_gate = nn.Sigmoid()(n3)

		local in_transform = nn.Tanh()(n4)

		local next_c           = nn.CAddTable()({
		    nn.CMulTable()({forget_gate, prev_c}),
		    nn.CMulTable()({in_gate,     in_transform})
		  })

		local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

			table.insert(outputs, next_c)
			table.insert(outputs, next_h)
	end

	if is_dec==1 then
		local last_h = outputs[#outputs]
		local pred = nn.Linear(rnn_size, output_size)(last_h)
		table.insert(outputs, pred)
	end

	return nn.gModule(inputs, outputs)
end

return LSTM