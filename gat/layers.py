import keras

class MultiHeadGraphAttention(keras.layers.Layer):
	'''
		Graph attention mechanism
		Cfr. P. Velickovic et al. (2017)
		Fixed version (v2) by S. Brody (2022)
	'''
	def __init__(
		self,
		units,
		num_heads=8,
		merge_type='concat',
		activation=keras.ops.gelu,
		dropout_rate=0, # Srivastava et al., 2014
		kernel_initializer='glorot_uniform',
		kernel_regularizer=None,
		random_gen=keras.random.SeedGenerator(),
		version=1, # 1: original (2017), 2: fixed version (2022) with coupled source/target weights, 3: fixed version with uncoupled weights
		use_bias=True,
		residual=False,
		repeat=False,
		use_embeddings=False,
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
		self.version = version
		self.residual = residual
		self.repeat = repeat
		self.use_embeddings = use_embeddings
		if merge_type == 'concat':
			self.output_dim = units * num_heads
		else:
			self.output_dim = units

	def build(self, input_shape):
		input_dim = input_shape[0][-1]
		#  kernel shape = (F, F'),
		# where F is the number of input features per node and F' is the number of output features per node
		self.kernel = []
		self.kernel.append(self.add_weight(
			shape=(input_dim, self.num_heads, self.units),
			initializer=self.kernel_initializer,
			regularizer=self.kernel_regularizer,
			name='kernel',
		))
		if self.version == 3:
			self.kernel.append(self.add_weight(
				shape=(input_dim, self.num_heads, self.units),
				initializer=self.kernel_initializer,
				regularizer=self.kernel_regularizer,
				name='kernel2',
			))
		if self.use_embeddings:
			self.kernel_embeddings = self.add_weight(
				shape=(1, self.num_heads),
				initializer=self.kernel_initializer,
				regularizer=self.kernel_regularizer,
				name='kernel_embeddings',
			)
		self.kernel_attention1 = self.add_weight(
			shape=(1, self.num_heads, self.units),
			initializer=self.kernel_initializer,
			regularizer=self.kernel_regularizer,
			name='kernel_attention1',
		)
		if self.version == 1:
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
		if self.use_embeddings:
			x, embeddings, edges = inputs
		else:
			x, edges = inputs
		targets, sources = edges[:, 1], edges[:, 0]

		n_nodes = keras.ops.shape(x)[-2]

		#  x shape = (N, F)
		#  edges shape = (E, 2)
		# where N is the total number of nodes, F is the number of input features for each node
		# and E is the total number of graph edges

		# linearly transform node states and apply dropout
		xp = []
		for n, kernel in enumerate(self.kernel):
			if training and self.dropout_rate > 0:
				if self.repeat:
					x_head = []
					for i in range(self.num_heads):
						xn = keras.random.dropout(x, self.dropout_rate, seed=self.random_gen)
						xn = keras.ops.tensordot(xn, keras.ops.take(kernel, i, axis=1), axes=1)
						x_head.append(keras.ops.expand_dims(xn, axis=-2))
					xn = keras.ops.concatenate(x_head, axis=-2)
				else:
					xn = keras.random.dropout(x, self.dropout_rate, seed=self.random_gen)
					xn = keras.ops.tensordot(xn, kernel, axes=1)
			else:
				xn = keras.ops.tensordot(x, kernel, axes=1)
			xp.append(xn)
			# shape = (N, H, F')

		if self.version >= 2 and self.use_bias:
			xbias = keras.ops.add(xp[0], self.bias_attention)
		else:
			xbias = xp[0]

		# (1) compute pair-wise attention scores
		if self.version == 1:
			f_t = keras.ops.sum(xbias * self.kernel_attention1, axis=-1)
			f_s = keras.ops.sum(xbias * self.kernel_attention2, axis=-1)
			f_t = keras.ops.take(f_t, targets, axis=0)
			f_s = keras.ops.take(f_s, sources, axis=0)
			scores = keras.ops.leaky_relu(f_t + f_s)
		else:
			if self.version == 3:
				f_t = keras.ops.take(xp[1], targets, axis=0)
			else:
				f_t = keras.ops.take(xbias, targets, axis=0)
			f_s = keras.ops.take(xbias, sources, axis=0)
			scores = keras.ops.leaky_relu(f_t + f_s)
			scores = keras.ops.sum(scores * self.kernel_attention1, axis=-1)
		# shape = (E, H)
		if self.use_embeddings:
			e_head = []
			for i in range(self.num_heads):
				e_head.append(keras.ops.expand_dims(embeddings, axis=-2))
			e_head = keras.ops.concatenate(e_head, axis=-2)
			e_t = keras.ops.take(e_head, targets, axis=0)
			e_s = keras.ops.take(e_head, sources, axis=0)
			scores += keras.ops.sum(e_t * e_s, axis=-1) * self.kernel_embeddings

		# (2) normalize attention scores
		scores = keras.ops.exp(scores - keras.ops.take(keras.ops.segment_max(scores, targets, n_nodes), targets, axis=0))
		scores /= keras.ops.take(keras.ops.segment_sum(scores, targets, n_nodes) + 1e-7, targets, axis=0)
		scores = scores[..., None]

		if training and self.dropout_rate > 0:
			scores = keras.random.dropout(scores, self.dropout_rate, seed=self.random_gen)
			xp[0] = keras.random.dropout(xp[0], self.dropout_rate, seed=self.random_gen)

		# (3) gather node states of neighbors, apply attention scores and aggregate
		out = scores * keras.ops.take(xp[0], sources, axis=0)
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

