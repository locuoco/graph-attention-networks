import keras

class MultiHeadGraphAttention(keras.layers.Layer):
	'''
		Graph attention mechanism
		Cfr. P. Velickovic et al. (2017)
		Fixed version (v2) by S. Brody (2021)
	'''
	def __init__(
		self,
		units,
		num_heads=8,
		merge_type='concat',
		activation=keras.ops.elu,
		dropout_rate=0,
		kernel_initializer='glorot_normal',
		kernel_regularizer=None,
		random_gen=keras.random.SeedGenerator(),
		use_bias=True,
		use_v2=False, # fixed version by Brody et al. (2021)
		residual=False,
		repeat=False,
		**kwargs,
	):
		super(MultiHeadGraphAttention, self).__init__(**kwargs)
		self.units = units
		self.num_heads = num_heads
		self.merge_type = merge_type
		self.activation = activation
		self.dropout_rate = dropout_rate
		self.kernel_initializer = keras.initializers.get(kernel_initializer)
		self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
		self.random_gen = random_gen
		self.use_bias = use_bias
		self.use_v2 = use_v2
		self.residual = residual
		self.repeat = repeat
		if merge_type == 'concat':
			self.output_dim = units * num_heads
		else:
			self.output_dim = units

	def build(self, input_shape):
		input_dim = input_shape[0][-1]
		#  kernel shape = (F, F'),
		# where F is the number of input features per node and F' is the number of output features per node
		self.kernel = self.add_weight(
			shape=(input_dim, self.units*self.num_heads),
			initializer=self.kernel_initializer,
			regularizer=self.kernel_regularizer,
			name='kernel',
		)
		self.kernel_attention1 = self.add_weight(
			shape=(1, self.num_heads, self.units),
			initializer=self.kernel_initializer,
			regularizer=self.kernel_regularizer,
			name='kernel_attention1',
		)
		if not self.use_v2:
			self.kernel_attention2 = self.add_weight(
				shape=(1, self.num_heads, self.units),
				initializer=self.kernel_initializer,
				regularizer=self.kernel_regularizer,
				name='kernel_attention2',
			)
		elif self.use_bias:
			self.bias_attention = self.add_weight(
				shape=(1, self.num_heads, self.units,),
				initializer='zeros',
				name='bias_attention',
			)
		if self.use_bias:
			self.bias = self.add_weight(
				shape=(self.output_dim,),
				initializer='zeros',
				name='bias',
			)
		if self.residual and input_dim != self.output_dim:
			self.residual_weights = True
			self.kernel_residual = self.add_weight(
				shape=(input_dim, self.output_dim),
				initializer=self.kernel_initializer,
				regularizer=self.kernel_regularizer,
				name='kernel_residual',
			)
		else:
			self.residual_weights = False
		super(MultiHeadGraphAttention, self).build(input_shape)
		self.built = True

	def call(self, inputs, training):
		output = self.call_sparse_edges_(inputs, training)
		if self.use_bias:
			output = keras.ops.add(output, self.bias)
		# activate and return node states
		return self.activation(output)

	def call_sparse_edges_(self, inputs, training):
		x, edges = inputs
		targets, sources = edges[:, 1], edges[:, 0]

		n_nodes = keras.ops.shape(x)[-2]

		#  x shape = (N, F)
		#  edges shape = (E, 2)
		# where N is the total number of nodes, F is the number of input features for each node
		# and E is the total number of graph edges

		# linearly transform node states and apply dropout
		if training and self.dropout_rate > 0:
			if self.repeat:
				kernel = keras.ops.reshape(self.kernel, (-1, self.num_heads, self.units))
				x_head = []
				for i in range(self.num_heads):
					xp = keras.random.dropout(x, self.dropout_rate, seed=self.random_gen)
					x_head.append(keras.ops.tensordot(xp, keras.ops.take(kernel, i, axis=1), axes=1))
				xp = keras.ops.concatenate(x_head, axis=-1)
			else:
				xp = keras.random.dropout(x, self.dropout_rate, seed=self.random_gen)
				xp = keras.ops.tensordot(xp, self.kernel, axes=1)
		else:
			xp = keras.ops.tensordot(x, self.kernel, axes=1)
		# shape = (N, H F'), where F' = self.units is the number of output features for each node
		xp = keras.ops.reshape(xp, (-1, self.num_heads, self.units))
		# shape = (N, H, F')

		if self.use_v2 and self.use_bias:
			xbias = keras.ops.add(xp, self.bias_attention)
		else:
			xbias = xp

		# (1) compute pair-wise attention scores
		if not self.use_v2:
			f_t = keras.ops.sum(xbias * self.kernel_attention1, axis=-1)
			f_s = keras.ops.sum(xbias * self.kernel_attention2, axis=-1)
			f_t = keras.ops.take(f_t, targets, axis=0)
			f_s = keras.ops.take(f_s, sources, axis=0)
			scores = keras.ops.leaky_relu(f_t + f_s)
		else:
			f_t = keras.ops.take(xbias, targets, axis=0)
			f_s = keras.ops.take(xbias, sources, axis=0)
			scores = keras.ops.leaky_relu(f_t + f_s)
			scores = keras.ops.sum(scores * self.kernel_attention1, axis=-1)
		# shape = (E, H)

		# (2) normalize attention scores
		scores = keras.ops.exp(scores - keras.ops.take(keras.ops.segment_max(scores, targets, n_nodes), targets, axis=0))
		scores /= keras.ops.take(keras.ops.segment_sum(scores, targets, n_nodes) + 1e-7, targets, axis=0)
		scores = scores[..., None]

		if training and self.dropout_rate > 0:
			scores = keras.random.dropout(scores, self.dropout_rate, seed=self.random_gen)
			xp = keras.random.dropout(xp, self.dropout_rate, seed=self.random_gen)

		# (3) gather node states of neighbors, apply attention scores and aggregate
		out = scores * keras.ops.take(xp, sources, axis=0)
		out = keras.ops.segment_sum(out, targets, n_nodes)
		# concatenate or average the node states from each head
		if self.merge_type == 'concat':
			out = keras.ops.reshape(out, (-1, self.num_heads * self.units))
		else:
			out = keras.ops.mean(out, axis=-2)
		# residual (skip) connections
		if self.residual:
			if self.residual_weights:
				out = keras.ops.add(out, keras.ops.tensordot(x, self.kernel_residual, axes=1))
			else:
				out = keras.ops.add(out, x)
		return out
