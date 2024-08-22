import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import gat.layers

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
			gat.layers.MultiHeadGraphAttention(hidden_units, num_heads) for _ in range(num_layers)
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
		self.attention_layer1 = gat.layers.MultiHeadGraphAttention(8, 8, dropout_rate=0.6)
		self.attention_layer2 = gat.layers.MultiHeadGraphAttention(output_dim, 1, dropout_rate=0.6)

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
