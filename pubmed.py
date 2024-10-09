from statistics import mean, stdev
import keras
import dgl

import gat.models
import src.losses

dataset = dgl.data.PubmedGraphDataset()
graph = dataset[0]
nodes_indices = keras.ops.convert_to_tensor(graph.nodes())
edges = keras.ops.transpose(keras.ops.convert_to_tensor(graph.edges()))

print('Edges shape:', edges.shape)

# get split masks
val_mask = keras.ops.convert_to_tensor(graph.ndata['val_mask'])
test_mask = keras.ops.convert_to_tensor(graph.ndata['test_mask'])
train_mask = keras.ops.convert_to_tensor(graph.ndata['train_mask'])

# get split indices
val_indices = nodes_indices[val_mask]
test_indices = nodes_indices[test_mask]
train_indices = nodes_indices[train_mask]

# get node features
features = keras.ops.convert_to_tensor(graph.ndata['feat'])

print('Node features shape:', features.shape)

# get ground-truth labels
labels = keras.ops.convert_to_tensor(graph.ndata['label'])

# get split ground-truths
val_labels = labels[val_mask]
test_labels = labels[test_mask]
train_labels = labels[train_mask]

# train and evaluate

# define hyper-parameters
output_dim = int(keras.ops.amax(labels))+1

num_epochs = 1000
batch_size = 512
learning_rate = 0.01

keras.utils.set_random_seed(1234)
random_gen = keras.random.SeedGenerator(1234)

weightsfile = './weights/pubmed.weights.h5'

iterations = 20
accs = []
for i in range(iterations):
	loss_fn = src.losses.SparseCategoricalCrossentropy(from_logits=True, label_smoothing=0.2)
	optimizer = keras.optimizers.Adam(learning_rate)
	accuracy_fn = keras.metrics.SparseCategoricalAccuracy(name='acc')
	early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)
	checkpoint = keras.callbacks.ModelCheckpoint(
		weightsfile,
		monitor='val_loss',
		save_best_only=True,
		save_weights_only=True,
	)

	# build model
	gat_model = gat.models.GraphAttentionNetworkTransductive2(
		features, edges, output_dim, random_gen=random_gen
	)

	# compile model
	gat_model.compile(loss=loss_fn, optimizer=optimizer, metrics=[accuracy_fn])

	gat_model.fit(
		x=train_indices,
		y=train_labels,
		validation_data=(val_indices, val_labels),
		batch_size=batch_size,
		epochs=num_epochs,
		callbacks=[early_stopping, checkpoint],
		verbose=0,
	)

	gat_model.load_weights(weightsfile) # restore best weights

	_, test_accuracy = gat_model.evaluate(x=test_indices, y=test_labels, verbose=0)
	print('--'*38 + f'\nTest Accuracy {i}: {test_accuracy*100:.1f}%')
	accs.append(test_accuracy)

print('--'*38 + f'\nTest Accuracy ({mean(accs)*100:.1f} +/- {stdev(accs)*100:.1f})%')










