import keras

import gat.layers

class DataGenerator(keras.utils.PyDataset):
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
		random_gen=keras.random.SeedGenerator(),
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
			random_gen=random_gen,
			repeat=True,
		)
		self.attention_layer2 = gat.layers.MultiHeadGraphAttention(
			output_dim,
			1,
			dropout_rate=0.6,
			kernel_regularizer=keras.regularizers.L2(2.5e-4),
			random_gen=random_gen,
		)

	def call(self, indices, training=False):
		x = self.attention_layer1((self.node_states, self.edges), training=training)
		outputs = self.attention_layer2((x, self.edges), training=training)
		return keras.ops.take(outputs, indices, axis=0)

class GraphAttentionNetworkTransductive2(keras.Model):
	# Variant of the previous model, used for Pubmed dataset
	def __init__(
		self,
		node_states,
		edges,
		output_dim,
		random_gen=keras.random.SeedGenerator(),
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
			random_gen=random_gen,
			repeat=True,
		)
		self.attention_layer2 = gat.layers.MultiHeadGraphAttention(
			output_dim,
			8,
			merge_type='avg',
			dropout_rate=0.5,
			kernel_regularizer=keras.regularizers.L2(5e-4),
			random_gen=random_gen,
			repeat=True,
		)

	def call(self, indices, training=False):
		x = self.attention_layer1((self.node_states, self.edges), training=training)
		outputs = self.attention_layer2((x, self.edges), training=training)
		return keras.ops.take(outputs, indices, axis=0)

class GraphAttentionNetworkInductive(keras.Model):
	def __init__(
		self,
		output_dim,
		random_gen=keras.random.SeedGenerator(),
		**kwargs,
	):
		super().__init__(**kwargs)
		self.attention_layer1 = gat.layers.MultiHeadGraphAttention(128, 4, random_gen=random_gen, residual=True)
		self.attention_layer2 = gat.layers.MultiHeadGraphAttention(128, 4, random_gen=random_gen, residual=True)
		self.attention_layer3 = gat.layers.MultiHeadGraphAttention(output_dim, 6, random_gen=random_gen, residual=True, merge_type='avg')

	def call(self, inputs, training=False):
		input_features, edges = inputs
		x = self.attention_layer1((input_features, edges), training=training)
		x = self.attention_layer2((x, edges), training=training)
		outputs = self.attention_layer3((x, edges), training=training)
		return outputs

