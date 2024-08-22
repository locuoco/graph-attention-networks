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
		kernel_regularizer=keras.regularizers.L2(5e-4),
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
		kernel_regularizer=keras.regularizers.L2(5e-4),
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

class GraphAttentionNetwork(keras.Model):
	def __init__(
		self,
		node_states,
		edges,
		hidden_units,
		num_heads,
		num_layers,
		output_dim,
		**kwargs,
	):
		super().__init__(**kwargs)
		self.node_states = node_states
		self.edges = edges
		self.preprocess = layers.Dense(hidden_units*num_heads, activation='relu')
		self.attention_layers = [
			MultiHeadGraphAttention(hidden_units, num_heads) for _ in range(num_layers)
		]
		self.output_layer = layers.Dense(output_dim)

	def set_graph(self, node_states, edges):
		# call this only during inference to change the underlying graph structure. Neural network weights will be preserved.
		# The number of input features must be the same as the old graph
		self.node_states = node_states
		self.edges = edges

	def call(self, inputs):
		node_states, edges = inputs
		x = self.preprocess(node_states)
		for attention_layer in self.attention_layers:
			x = attention_layer([x, edges]) + x
		outputs = self.output_layer(x)
		return outputs

	def train_step(self, data):
		indices, labels = data

		with tf.GradientTape() as tape:
			# forward pass
			outputs = self([self.node_states, self.edges])
			# compute loss
			loss = self.compiled_loss(labels, tf.gather(outputs, indices))
			# add regularization losses
			loss += tf.reduce_sum(self.losses)
		# compute gradients
		grads = tape.gradient(loss, self.trainable_weights)
		# apply gradients (update weights)
		self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
		# update metric(s)
		self.compiled_metrics.update_state(labels, tf.gather(outputs, indices))

		return {m.name: m.result() for m in self.metrics}

	def predict_step(self, data):
		indices = data
		# forward pass
		outputs = self([self.node_states, self.edges])
		# compute probabilities
		return tf.nn.softmax(tf.gather(outputs, indices))

	def test_step(self, data):
		indices, labels = data
		# forward pass
		outputs = self([self.node_states, self.edges])
		# compute loss
		loss = self.compiled_loss(labels, tf.gather(outputs, indices))
		# update metric(s)
		self.compiled_metrics.update_state(labels, tf.gather(outputs, indices))

		return {m.name: m.result() for m in self.metrics}

class GraphAttentionNetworkTransductive(keras.Model):
	def __init__(
		self,
		node_states,
		edges,
		output_dim,
		**kwargs,
	):
		super().__init__(**kwargs)
		self.node_states = node_states
		self.edges = edges
		self.attention_layer1 = MultiHeadGraphAttention(8, 8, dropout_rate=0.6)
		self.attention_layer2 = MultiHeadGraphAttention(output_dim, 1, dropout_rate=0.6)

	def call(self, inputs, training):
		node_states, edges = inputs
		x = self.attention_layer1([node_states, edges], training=training)
		outputs = self.attention_layer2([x, edges], training=training)
		return outputs

	def train_step(self, data):
		indices, labels = data

		with tf.GradientTape() as tape:
			# forward pass
			outputs = self([self.node_states, self.edges], training=True)
			# compute loss
			loss = self.compiled_loss(labels, tf.gather(outputs, indices))
			# add regularization losses
			loss += tf.reduce_sum(self.losses)
		# compute gradients
		grads = tape.gradient(loss, self.trainable_weights)
		# apply gradients (update weights)
		self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
		# update metric(s)
		self.compiled_metrics.update_state(labels, tf.gather(outputs, indices))

		return {m.name: m.result() for m in self.metrics}

	def predict_step(self, data):
		indices = data
		# forward pass
		outputs = self([self.node_states, self.edges], training=False)
		# compute probabilities
		return tf.nn.softmax(tf.gather(outputs, indices))

	def test_step(self, data):
		indices, labels = data
		# forward pass
		outputs = self([self.node_states, self.edges], training=False)
		# compute loss
		loss = self.compiled_loss(labels, tf.gather(outputs, indices))
		# update metric(s)
		self.compiled_metrics.update_state(labels, tf.gather(outputs, indices))

		return {m.name: m.result() for m in self.metrics}

