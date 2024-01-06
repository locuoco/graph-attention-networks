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
		kernel_initializer='glorot_uniform',
		kernel_regularizer=None,
		**kwargs,
	):
		super().__init__(**kwargs)
		self.units = units
		self.kernel_initializer = keras.initializers.get(kernel_initializer)
		self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

	def build(self, input_shape):
		self.kernel = self.add_weight(
			shape=(input_shape[0][-1], self.units),
			trainable=True,
			initializer=self.kernel_initializer,
			regularizer=self.kernel_regularizer,
			name='kernel',
		)
		self.kernel_attention = self.add_weight(
			shape=(self.units*2, 1),
			trainable=True,
			initializer=self.kernel_initializer,
			regularizer=self.kernel_regularizer,
			name='kernel_attention',
		)
		self.built = True

	def call(self, inputs):
		node_states, edges = inputs

		# linearly transform node states
		node_states_transformed = tf.matmul(node_states, self.kernel)

		# (1) compute pair-wise attention scores
		node_states_expanded = tf.gather(node_states_transformed, edges)
		node_states_expanded = tf.reshape(
			node_states_expanded, (tf.shape(edges)[0], -1)
		)
		attention_scores = tf.nn.leaky_relu(
			tf.matmul(node_states_expanded, self.kernel_attention)
		)
		attention_scores = tf.squeeze(attention_scores, -1)

		# (2) normalize attention scores
		attention_scores = tf.math.exp(tf.clip_by_value(attention_scores, -2, 2))
		attention_scores_sum = tf.math.unsorted_segment_sum(
			data=attention_scores,
			segment_ids=edges[:, 0],
			num_segments=tf.reduce_max(edges[:, 0]) +1,
		)
		attention_scores_sum = tf.repeat(
			attention_scores_sum, tf.math.bincount(tf.cast(edges[:, 0], 'int32'))
		)
		attention_scores_norm = attention_scores / attention_scores_sum

		# (3) gather node states of neighbors, apply attention scores and aggregate
		node_states_neighbors = tf.gather(node_states_transformed, edges[:, 1])
		out = tf.math.unsorted_segment_sum(
			data=node_states_neighbors*attention_scores_norm[:, tf.newaxis],
			segment_ids=edges[:, 0],
			num_segments=tf.shape(node_states)[0],
		)
		return out

class MultiHeadGraphAttention(layers.Layer):
	def __init__(self, units, num_heads=8, merge_type='concat', **kwargs):
		super().__init__(**kwargs)
		self.num_heads = num_heads
		self.merge_type = merge_type
		self.attention_layers = [GraphAttention(units) for _ in range(num_heads)]

	def call(self, inputs):
		atom_features, pair_indices = inputs

		# obtain outputs from each attention head
		outputs = [
			attention_layer([atom_features, pair_indices])
			for attention_layer in self.attention_layers
		]
		# concatenate or average the node states from each head
		if self.merge_type == 'concat':
			outputs = tf.concat(outputs, axis=-1)
		else:
			outputs = tf.reduce_mean(tf.stack(outputs, axis=-1), axis=-1)
		# activate and return node states
		return tf.nn.relu(outputs)

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
		# compute gradients
		grads = tape.gradient(loss, self.trainable_weighs)
		# apply gradients (update weights)
		optimizer.apply_gradients(zip(grads, self.trainable_weights))
		# update metric(s)
		self.compiled_metrics.update_state(labels, tf.gather(output, indices))

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

