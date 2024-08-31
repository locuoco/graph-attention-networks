import keras
import dgl

import gat.models

dataset = dgl.data.CoraGraphDataset()
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

# train and evalate

# define hyper-parameters
output_dim = int(keras.ops.amax(labels))+1

num_epochs = 1000
batch_size = 512
learning_rate = 0.005

keras.utils.set_random_seed(1234)
random_gen = keras.random.SeedGenerator(1234)

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam(learning_rate)
accuracy_fn = keras.metrics.SparseCategoricalAccuracy(name='acc')
early_stopping = keras.callbacks.EarlyStopping(
	monitor='val_loss',
	patience=100,
	restore_best_weights=True
)

# build model
gat_model = gat.models.GraphAttentionNetworkTransductive(
	features, edges, output_dim, random_gen
)

# compile model
gat_model.compile(loss=loss_fn, optimizer=optimizer, metrics=[accuracy_fn])

gat_model.fit(
	x=train_indices,
	y=train_labels,
	validation_data=(val_indices, val_labels),
	batch_size=batch_size,
	epochs=num_epochs,
	callbacks=[early_stopping],
	verbose=2,
)

_, test_accuracy = gat_model.evaluate(x=test_indices, y=test_labels, verbose=0)

print('--'*38 + f'\nTest Accuracy {test_accuracy*100:.1f}%')










