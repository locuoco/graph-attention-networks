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
			use_v2=True,
			repeat=True,
		)
		self.attention_layer2 = gat.layers.MultiHeadGraphAttention(
			output_dim,
			1,
			dropout_rate=0.6,
			kernel_regularizer=keras.regularizers.L2(2.5e-4),
			random_gen=random_gen,
			use_v2=True,
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
			activation=keras.ops.elu,
			dropout_rate=0.6,
			kernel_regularizer=keras.regularizers.L2(1e-3),
			random_gen=random_gen,
			use_v2=True,
			repeat=True,
		)
		self.attention_layer2 = gat.layers.MultiHeadGraphAttention(
			output_dim,
			8,
			merge_type='avg',
			activation=keras.ops.elu,
			dropout_rate=0.5,
			random_gen=random_gen,
			use_v2=True,
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
		self.random_gen = random_gen
		self.head_layer = keras.layers.Dense(1024)
		self.hidden_layers = []
		for _ in range(4):
			layer = {}
			layer['normg'] = keras.layers.LayerNormalization()
			layer['gat'] = gat.layers.MultiHeadGraphAttention(256, 4, dropout_rate=0.2, random_gen=random_gen, use_v2=True, residual=True)
			layer['normd'] = keras.layers.LayerNormalization()
			layer['dense'] = keras.layers.Dense(4096, activation=keras.ops.gelu)
			layer['add'] = keras.layers.Add()
			self.hidden_layers.append(layer)
		self.tail_layer = keras.layers.Dense(output_dim)

	def call(self, inputs, training=False):
		input_features, edges = inputs
		x = self.head_layer(input_features)
		for layer in self.hidden_layers:
			x = layer['normg'](x)
			x = layer['gat']((x, edges), training=training)
			xr = layer['normd'](x)
			if training:
				x = keras.random.dropout(xr, rate=0.2, seed=self.random_gen)
			else:
				x = xr
			x = layer['dense'](x)
			x = layer['add'](keras.ops.split(x, 4, axis=-1)) + xr # residual
		outputs = self.tail_layer(x)
		return outputs

class GraphAttentionNetworkTransductive3(keras.Model):
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
		self.random_gen = random_gen
		self.head_layer = keras.layers.Dense(512)
		self.hidden_layers = []
		for _ in range(4):
			layer = {}
			layer['normg'] = keras.layers.LayerNormalization()
			layer['gat'] = gat.layers.MultiHeadGraphAttention(128, 4, dropout_rate=0.7, random_gen=random_gen, use_v2=True, residual=True)
			layer['normd'] = keras.layers.LayerNormalization()
			layer['dense'] = keras.layers.Dense(2048, activation=keras.ops.gelu)
			layer['add'] = keras.layers.Add()
			self.hidden_layers.append(layer)
		self.tail_layer = keras.layers.Dense(output_dim)

	def call(self, indices, training=False):
		x = self.head_layer(self.node_states)
		for layer in self.hidden_layers:
			x = layer['normg'](x)
			x = layer['gat']((x, self.edges), training=training)
			xr = layer['normd'](x)
			if training:
				x = keras.random.dropout(xr, rate=0.7, seed=self.random_gen)
			else:
				x = xr
			x = layer['dense'](x)
			x = layer['add'](keras.ops.split(x, 4, axis=-1)) + xr # residual
		outputs = self.tail_layer(x)
		return keras.ops.take(outputs, indices, axis=0)

