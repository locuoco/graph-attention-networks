import os
import pickle

# Set environment variables for JAX memory limits
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'

import networkx as nx
import keras
import dgl

import gat.models
import src.optimizers
import src.losses

load_last_weights = False
continue_training = False
initial_epoch = 0

datafile = './data/closeness.pkl'

if not os.path.isfile(datafile):
	print('Calculating labels and embeddings... This may take a while')

	val_dataset = dgl.data.PPIDataset(mode='valid')
	test_dataset = dgl.data.PPIDataset(mode='test')
	train_dataset = dgl.data.PPIDataset(mode='train')
	val_graphs = []
	test_graphs = []
	train_graphs = []
	val_labels = []
	test_labels = []
	train_labels = []
	mode_datasets = [val_dataset, test_dataset, train_dataset]
	mode_graphs = [val_graphs, test_graphs, train_graphs]
	mode_labels = [val_labels, test_labels, train_labels]

	for i, dataset in enumerate(mode_datasets):
		for graph in dataset:
			# get closeness centrality measures as node labels
			g = dgl.to_networkx(graph)
			label = nx.closeness_centrality(g)
			label = keras.ops.convert_to_tensor([label[node] for node in g.nodes])
			label = keras.ops.expand_dims(label, axis=-1)
			mode_labels[i].append(label)

			# get graph laplacian eigenmap (spectral positional node encodings)
			eigvecs, eigvals = dgl.lap_pe(graph, 100, return_eigval=True)
			eigvecs = keras.ops.convert_to_tensor(eigvecs)
			eigvals = keras.ops.convert_to_tensor(eigvals)
			embeddings = []
			for k, eigval in enumerate(eigvals):
				if eigval > 1e-10 and len(embeddings) < 20:
					embeddings.append(eigvecs[:, k])
			embeddings = keras.ops.transpose(keras.ops.convert_to_tensor(embeddings)) * keras.ops.sqrt(graph.num_nodes())
			features = keras.ops.abs(embeddings)

			# get edges
			edges = keras.ops.transpose(keras.ops.convert_to_tensor(graph.edges(), dtype='int32'))
			mode_graphs[i].append((features, embeddings, edges))

	with open(datafile, 'wb') as f:
		pickle.dump((mode_labels, mode_graphs), f, pickle.HIGHEST_PROTOCOL)
else:
	try:
		with open(datafile, 'rb') as f:
			mode_labels, mode_graphs = pickle.load(f)
	except:
		print('Cannot load ', datafile, ', try to delete it before running the script.', sep='')
		raise
	val_labels, test_labels, train_labels = mode_labels
	val_graphs, test_graphs, train_graphs = mode_graphs

# train and evaluate

# define hyper-parameters
output_dim = 1

num_epochs = 5000
#batch_size = 1 # number of graphs per batch
learning_rate = 0.001

keras.utils.set_random_seed(1234)
random_gen = keras.random.SeedGenerator(1234)

mase_fn = src.losses.MeanAbsoluteScaledError(name='mase')
mae_fn = keras.losses.MeanAbsoluteError(name='mae')
mse_fn = keras.losses.MeanSquaredError(name='mse')
optimizer = src.optimizers.Adan(learning_rate)
early_stopping = keras.callbacks.EarlyStopping(
	patience=300,
	restore_best_weights=True
)

# build model
gat_model = gat.models.GraphAttentionNetworkInductive(
	output_dim,
	units_per_head=32,
	num_heads=16,
	num_layers=8,
	dropout_rate=0,
	use_layer_norm=True,
	use_dense=False,
	use_embeddings=True,
	random_gen=random_gen
)

# compile model
gat_model.compile(loss=mae_fn, optimizer=optimizer, metrics=[mase_fn, mse_fn])

weightsfile = './weights/closeness.weights.h5'

if load_last_weights and os.path.isfile(weightsfile):
	gat_model(train_graphs[0]) # force model building
	gat_model.load_weights(weightsfile)

val_generator = gat.models.DataGenerator(val_graphs, val_labels)
test_generator = gat.models.DataGenerator(test_graphs, test_labels)
train_generator = gat.models.DataGenerator(train_graphs, train_labels)

if not load_last_weights or continue_training:
	gat_model.fit(
		train_generator,
		validation_data=val_generator,
		epochs=num_epochs,
		callbacks=[early_stopping],
		verbose=2,
		initial_epoch=initial_epoch,
	)

test_mae, test_mase, test_mse = gat_model.evaluate(test_generator, verbose=0)

gat_model.save_weights(weightsfile)

print('--'*38 + f'\nTest MAE: {test_mae:.4f}, MASE: {test_mase:.4f}, MSE: {test_mse:.4e}')










