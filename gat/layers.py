import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class GraphAttention(layers.Layer):
	'''
		Graph attention mechanism
		Cfr. P. Velickovic et al. (2017)
	'''
	def __init__(
		self,
		units,
		dropout_rate=0,
		kernel_initializer='glorot_normal',
		kernel_regularizer=None,
		**kwargs,
	):
		super().__init__(**kwargs)
		self.units = units
		self.dropout_rate = dropout_rate
		self.kernel_initializer = keras.initializers.get(kernel_initializer)
		self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

	def build(self, input_shape):
		#  kernel shape = (F, F'),
		# where F is the number of input features per node and F' is the number of output features per node
		self.kernel = self.add_weight(
			shape=(input_shape[0][-1], self.units),
			trainable=True,
			initializer=self.kernel_initializer,
			regularizer=self.kernel_regularizer,
			name='kernel',
		)
		self.bias = self.add_weight(
			shape=(self.units,),
			trainable=True,
			initializer='zeros',
			name='bias',
		)
		self.kernel_attention = self.add_weight(
			shape=(self.units*2,),
			trainable=True,
			initializer=self.kernel_initializer,
			regularizer=self.kernel_regularizer,
			name='kernel_attention',
		)
		self.built = True

	def call(self, inputs, training):
		#  node_states shape = (N, F)
		#  edges shape = (E, 2)
		# where N is the total number of nodes, F is the number of input features for each node
		# and E is the total number of graph edges
		node_states, edges = inputs

		if training:
			node_states = tf.nn.dropout(node_states, self.dropout_rate)

		# linearly transform node states
		node_states_transformed = tf.matmul(node_states, self.kernel)
		# shape = (N, F'), where F' = self.units is the number of output features for each node

		# (1) compute pair-wise attention scores
		node_states_expanded = tf.gather(node_states_transformed, edges)
		# shape = (E, 2, F')
		node_states_expanded = tf.reshape(
			node_states_expanded, (tf.shape(edges)[0], -1)
		)
		# shape = (E, 2F')
		attention_scores = tf.nn.leaky_relu(
			tf.tensordot(node_states_expanded, self.kernel_attention, 1)
		)
		# shape = (E,)

		# (2) normalize attention scores
		attention_scores = tf.math.exp(attention_scores)
		attention_scores_sum = tf.math.unsorted_segment_sum(
			data=attention_scores,
			segment_ids=edges[:, 0],
			num_segments=tf.reduce_max(edges[:, 0])+1,
		)
		# shape = (N,)
		attention_scores_sum = tf.repeat(
			attention_scores_sum, tf.math.bincount(tf.cast(edges[:, 0], 'int32'))
		)
		# shape = (E,)
		attention_scores_norm = attention_scores / attention_scores_sum

		if training:
			attention_scores_norm = tf.nn.dropout(attention_scores_norm, self.dropout_rate)

		if training:
			node_states_transformed = tf.nn.dropout(node_states_transformed, self.dropout_rate)

		# (3) gather node states of neighbors, apply attention scores and aggregate
		node_states_neighbors = tf.gather(node_states_transformed, edges[:, 1])
		# shape = (E, F')
		out = tf.math.unsorted_segment_sum(
			data=node_states_neighbors*attention_scores_norm[:, tf.newaxis],
			segment_ids=edges[:, 0],
			num_segments=tf.shape(node_states)[0],
		)
		# shape = (N, F')
		return tf.nn.bias_add(out, self.bias)

class MultiHeadGraphAttention(layers.Layer):
	def __init__(
		self,
		units,
		num_heads=8,
		merge_type='concat',
		activation=tf.nn.elu,
		dropout_rate=0,
		kernel_initializer='glorot_normal',
		kernel_regularizer=None,
		**kwargs,
	):
		super().__init__(**kwargs)
		self.num_heads = num_heads
		self.merge_type = merge_type
		self.attention_layers = [GraphAttention(units, dropout_rate, kernel_initializer, kernel_regularizer) for _ in range(num_heads)]
		self.activation = activation

	def call(self, inputs, training):
		atom_features, pair_indices = inputs

		# obtain outputs from each attention head
		outputs = [
			attention_layer([atom_features, pair_indices], training=training)
			for attention_layer in self.attention_layers
		]
		# concatenate or average the node states from each head
		if self.merge_type == 'concat':
			outputs = tf.concat(outputs, axis=-1)
		else:
			outputs = tf.reduce_mean(tf.stack(outputs, axis=-1), axis=-1)
		# activate and return node states
		return self.activation(outputs)

