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
			version=2,
			repeat=True,
		)
		self.attention_layer2 = gat.layers.MultiHeadGraphAttention(
			output_dim,
			1,
			dropout_rate=0.6,
			kernel_regularizer=keras.regularizers.L2(2.5e-4),
			random_gen=random_gen,
			version=2,
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
			version=2,
			repeat=True,
		)
		self.attention_layer2 = gat.layers.MultiHeadGraphAttention(
			output_dim,
			8,
			merge_type='avg',
			activation=keras.ops.elu,
			dropout_rate=0.5,
			random_gen=random_gen,
			version=2,
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
		units_per_head=256,
		num_heads=4,
		num_layers=4,
		dropout_rate=0.2,
		use_layer_norm=True,
		use_dense=True,
		residual=True,
		input_feats_as_embeddings=False,
		version=2,
		random_gen=keras.random.SeedGenerator(),
		**kwargs,
	):
		super(GraphAttentionNetworkInductive, self).__init__(**kwargs)
		self.dropout_rate = dropout_rate
		self.hidden_units = units_per_head*num_heads
		self.use_layer_norm = use_layer_norm
		self.use_dense = use_dense
		self.residual = residual
		self.input_feats_as_embeddings = input_feats_as_embeddings
		self.random_gen = random_gen
		self.head_layer = keras.layers.Dense(units_per_head*num_heads)
		self.hidden_layers = []
		for _ in range(num_layers):
			layer = {}
			if use_layer_norm:
				layer['normg'] = keras.layers.LayerNormalization()
			layer['gat'] = gat.layers.MultiHeadGraphAttention(
				units_per_head,
				num_heads,
				dropout_rate=dropout_rate,
				random_gen=random_gen,
				version=version,
				residual=residual,
				use_embeddings=input_feats_as_embeddings,
			)
			if use_dense:
				if use_layer_norm:
					layer['normd'] = keras.layers.LayerNormalization()
				layer['dense'] = keras.layers.Dense(units_per_head*num_heads*4, activation=keras.ops.gelu)
				layer['add'] = keras.layers.Add()
			self.hidden_layers.append(layer)
		self.tail_layer = keras.layers.Dense(output_dim)

	def call(self, inputs, training=False):
		input_features, edges = inputs
		x = self.head_layer(input_features)
		for layer in self.hidden_layers:
			if self.use_layer_norm:
				x = layer['normg'](x)
			if self.input_feats_as_embeddings:
				x = layer['gat']((x, input_features, edges), training=training)
			else:
				x = layer['gat']((x, edges), training=training)
			if self.use_dense:
				if self.use_layer_norm:
					xr = layer['normd'](x)
				else:
					xr = x
				if training:
					x = keras.random.dropout(xr, self.dropout_rate, seed=self.random_gen)
				else:
					x = xr
				x = layer['dense'](x)
				x = layer['add'](keras.ops.split(x, 4, axis=-1))
				if self.residual:
					x = x + xr
		outputs = self.tail_layer(x)
		return outputs

