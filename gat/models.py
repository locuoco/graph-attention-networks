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
			dropout_rate=0.6,
			kernel_regularizer=keras.regularizers.L2(5e-4),
			random_gen=random_gen,
			use_v2=True,
			residual=True,
			repeat=True,
		)
		self.attention_layer2 = gat.layers.MultiHeadGraphAttention(
			output_dim,
			8,
			merge_type='avg',
			dropout_rate=0.5,
			kernel_regularizer=keras.regularizers.L2(5e-4),
			random_gen=random_gen,
			use_v2=True,
			residual=True,
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
		self.head_layer = keras.layers.Dense(512)
		self.norm1 = keras.layers.LayerNormalization()
		self.attention_layer1 = gat.layers.MultiHeadGraphAttention(128, 4, random_gen=random_gen, residual=True)
		self.norm2 = keras.layers.LayerNormalization()
		self.attention_layer2 = gat.layers.MultiHeadGraphAttention(128, 4, random_gen=random_gen, residual=True)
		self.norm3 = keras.layers.LayerNormalization()
		self.attention_layer3 = gat.layers.MultiHeadGraphAttention(128, 4, random_gen=random_gen, residual=True)
		self.tail_layer = keras.layers.Dense(output_dim)

	def call(self, inputs, training=False):
		input_features, edges = inputs
		x = self.head_layer(input_features)
		x = self.norm1(x)
		x = self.attention_layer1((x, edges), training=training)
		x = self.norm2(x)
		x = self.attention_layer2((x, edges), training=training)
		x = self.norm3(x)
		x = self.attention_layer3((x, edges), training=training)
		outputs = self.tail_layer(x)
		return outputs

