import tensorflow as tf
import keras
from keras import layers

class MultiHeadGraphAttention(layers.Layer):
	'''
		Graph attention mechanism
		Cfr. P. Velickovic et al. (2017)
	'''
	def __init__(
		self,
		units,
		num_heads=8,
		merge_type='concat',
		activation=tf.nn.elu,
		dropout_rate=0,
		kernel_initializer='glorot_normal',
		kernel_regularizer=None,
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
		self.kernel_attention2 = self.add_weight(
			shape=(1, self.num_heads, self.units),
			initializer=self.kernel_initializer,
			regularizer=self.kernel_regularizer,
			name='kernel_attention2',
		)
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
		# activate and return node states
		return self.activation(output + self.bias)

	def call_sparse_edges_(self, inputs, training):
		x, edges = inputs
		targets, sources = edges[:, 1], edges[:, 0]

		n_nodes = tf.shape(x)[-2]

		#  x shape = (N, F)
		#  edges shape = (E, 2)
		# where N is the total number of nodes, F is the number of input features for each node
		# and E is the total number of graph edges

		# linearly transform node states and apply dropout
		if training and self.dropout_rate > 0:
			if self.repeat:
				kernel = tf.reshape(self.kernel, (-1, self.num_heads, self.units))
				x_head = []
				for i in range(self.num_heads):
					xp = tf.nn.dropout(x, self.dropout_rate)
					x_head.append(tf.tensordot(xp, tf.gather(kernel, i, axis=1), axes=1))
				xp = tf.concat(x_head, axis=-1)
			else:
				xp = tf.nn.dropout(x, self.dropout_rate)
				xp = tf.tensordot(xp, self.kernel, axes=1)
		else:
			xp = tf.tensordot(x, self.kernel, axes=1)
		# shape = (N, H F'), where F' = self.units is the number of output features for each node

		xp = tf.reshape(xp, (-1, self.num_heads, self.units))
		# shape = (N, H, F')

		# (1) compute pair-wise attention scores
		f_t = tf.reduce_sum(xp * self.kernel_attention1, -1)
		f_s = tf.reduce_sum(xp * self.kernel_attention2, -1)
		f_t = tf.gather(f_t, targets)
		f_s = tf.gather(f_s, sources)
		scores = tf.nn.leaky_relu(f_t + f_s)
		# shape = (E, H, F')

		# (2) normalize attention scores
		scores = tf.math.exp(scores - tf.gather(tf.math.unsorted_segment_max(scores, targets, n_nodes), targets))
		# shape = (N,)
		scores /= tf.gather(tf.math.unsorted_segment_sum(scores, targets, n_nodes) + 1e-7, targets)
		scores = scores[..., None]

		if training and self.dropout_rate > 0:
			scores = tf.nn.dropout(scores, self.dropout_rate)
			xp = tf.nn.dropout(xp, self.dropout_rate)

		# (3) gather node states of neighbors, apply attention scores and aggregate
		out = scores * tf.gather(xp, sources)
		# shape = (E, F')
		out = tf.math.unsorted_segment_sum(out, targets, n_nodes)
		# concatenate or average the node states from each head
		if self.merge_type == 'concat':
			out = tf.reshape(out, (-1, self.num_heads * self.units))
		else:
			out = tf.reduce_mean(out, axis=-2)
		# residual (skip) connections
		if self.residual:
			if self.residual_weights:
				out = tf.add(out, tf.tensordot(x, self.kernel_residual, axes=1))
			else:
				out = tf.add(out, x)
		return out
