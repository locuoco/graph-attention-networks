import tensorflow as tf
import keras
from keras import layers
from keras.utils import Sequence

import gat.layers

class DataGenerator(Sequence):
	def __init__(self, data, labels, **kwargs):
		super().__init__(**kwargs)
		self.data = data
		self.labels = labels

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		return self.data[index], self.labels[index]

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
		self.attention_layer1 = gat.layers.MultiHeadGraphAttention(
			8,
			8,
			dropout_rate=0.6,
			kernel_regularizer=keras.regularizers.L2(2.5e-4),
			repeat=True,
		)
		self.attention_layer2 = gat.layers.MultiHeadGraphAttention(
			output_dim,
			1,
			dropout_rate=0.6,
			kernel_regularizer=keras.regularizers.L2(2.5e-4),
			repeat=True,
		)

	def build(self, input_shape):
		self.built = True

	def call(self, inputs, training):
		node_states, edges = inputs
		x = self.attention_layer1((node_states, edges), training=training)
		outputs = self.attention_layer2((x, edges), training=training)
		return outputs

	def train_step(self, data):
		indices, labels = data

		with tf.GradientTape() as tape:
			# forward pass
			outputs = tf.gather(self((self.node_states, self.edges), training=True), indices)
			# compute loss
			loss = self.compute_loss(y=labels, y_pred=outputs)
			# add regularization losses
			loss += tf.reduce_sum(self.losses)
		# compute gradients
		grads = tape.gradient(loss, self.trainable_weights)
		# apply gradients (update weights)
		self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
		# update metric(s)
		for metric in self.metrics:
			if metric.name == 'loss':
				metric.update_state(loss)
			else:
				metric.update_state(labels, outputs)

		return {m.name: m.result() for m in self.metrics}

	def predict_step(self, data):
		indices = data
		# forward pass
		outputs = tf.gather(self((self.node_states, self.edges), training=False), indices)
		# compute probabilities
		return tf.nn.softmax(outputs)

	def test_step(self, data):
		indices, labels = data
		# forward pass
		outputs = tf.gather(self((self.node_states, self.edges), training=False), indices)
		# compute loss
		loss = self.compute_loss(y=labels, y_pred=outputs)
		# update metric(s)
		for metric in self.metrics:
			if metric.name == 'loss':
				metric.update_state(loss)
			else:
				metric.update_state(labels, outputs)

		return {m.name: m.result() for m in self.metrics}

class GraphAttentionNetworkTransductive2(keras.Model):
	# Variant of the previous model, used for Pubmed dataset
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
		self.attention_layer1 = gat.layers.MultiHeadGraphAttention(
			8,
			8,
			dropout_rate=0.5,
			kernel_regularizer=keras.regularizers.L2(5e-4),
			repeat=True,
		)
		self.attention_layer2 = gat.layers.MultiHeadGraphAttention(
			output_dim,
			8,
			merge_type='avg',
			dropout_rate=0.5,
			kernel_regularizer=keras.regularizers.L2(5e-4),
			repeat=True,
		)

	def build(self, input_shape):
		self.built = True

	def call(self, inputs, training):
		node_states, edges = inputs
		x = self.attention_layer1((node_states, edges), training=training)
		outputs = self.attention_layer2((x, edges), training=training)
		return outputs

	def train_step(self, data):
		indices, labels = data

		with tf.GradientTape() as tape:
			# forward pass
			outputs = tf.gather(self((self.node_states, self.edges), training=True), indices)
			# compute loss
			loss = self.compute_loss(y=labels, y_pred=outputs)
			# add regularization losses
			loss += tf.reduce_sum(self.losses)
		# compute gradients
		grads = tape.gradient(loss, self.trainable_weights)
		# apply gradients (update weights)
		self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
		# update metric(s)
		for metric in self.metrics:
			if metric.name == 'loss':
				metric.update_state(loss)
			else:
				metric.update_state(labels, outputs)

		return {m.name: m.result() for m in self.metrics}

	def predict_step(self, data):
		indices = data
		# forward pass
		outputs = tf.gather(self((self.node_states, self.edges), training=False), indices)
		# compute probabilities
		return tf.nn.softmax(outputs)

	def test_step(self, data):
		indices, labels = data
		# forward pass
		outputs = tf.gather(self((self.node_states, self.edges), training=False), indices)
		# compute loss
		loss = self.compute_loss(y=labels, y_pred=outputs)
		# update metric(s)
		for metric in self.metrics:
			if metric.name == 'loss':
				metric.update_state(loss)
			else:
				metric.update_state(labels, outputs)

		return {m.name: m.result() for m in self.metrics}

class GraphAttentionNetworkInductive(keras.Model):
	def __init__(
		self,
		output_dim,
		**kwargs,
	):
		super().__init__(**kwargs)
		self.attention_layer1 = gat.layers.MultiHeadGraphAttention(128, 4, residual=True)
		self.attention_layer2 = gat.layers.MultiHeadGraphAttention(128, 4, residual=True)
		self.attention_layer3 = gat.layers.MultiHeadGraphAttention(output_dim, 6, merge_type='avg', residual=True)

	def build(self, input_shape):
		self.built = True

	def call(self, inputs, training=False):
		input_features, edges = inputs
		x = self.attention_layer1((input_features, edges), training=training)
		x = self.attention_layer2((x, edges), training=training)
		outputs = self.attention_layer3((x, edges), training=training)
		return outputs

	def train_step(self, data):
		graph, labels = data

		with tf.GradientTape() as tape:
			# forward pass
			outputs = self(graph, training=True)
			# compute loss
			loss = self.compute_loss(y=labels, y_pred=outputs)
			# add regularization losses
			loss += tf.reduce_sum(self.losses)
		# compute gradients
		grads = tape.gradient(loss, self.trainable_weights)
		# apply gradients (update weights)
		self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
		# update metric(s)
		for metric in self.metrics:
			if metric.name == 'loss':
				metric.update_state(loss)
			else:
				metric.update_state(labels, outputs)

		return {m.name: m.result() for m in self.metrics}

	def predict_step(self, graph):
		# forward pass
		outputs = self(graph, training=False)
		# compute probabilities
		return tf.math.sigmoid(outputs)

	def test_step(self, data):
		graph, labels = data
		# forward pass
		outputs = self(graph, training=False)
		# compute loss
		loss = self.compute_loss(y=labels, y_pred=outputs)
		# update metric(s)
		for metric in self.metrics:
			if metric.name == 'loss':
				metric.update_state(loss)
			else:
				metric.update_state(labels, outputs)

		return {m.name: m.result() for m in self.metrics}

