import tensorflow as tf
import keras
import dgl

import gat.models

dataset = dgl.data.CoraGraphDataset()
graph = dataset[0]
nodes_indices = graph.nodes()
edges = tf.transpose(tf.convert_to_tensor(graph.edges()))

print('Edges shape:', edges.shape)

# get split masks
val_mask = graph.ndata['val_mask']
test_mask = graph.ndata['test_mask']
train_mask = graph.ndata['train_mask']

# get split indices
val_indices = tf.boolean_mask(nodes_indices, val_mask)
test_indices = tf.boolean_mask(nodes_indices, test_mask)
train_indices = tf.boolean_mask(nodes_indices, train_mask)

# get node features
features = graph.ndata['feat']

print('Node features shape:', features.shape)

# get ground-truth labels
labels = graph.ndata['label']

# get split ground-truths
val_labels = tf.boolean_mask(labels, val_mask)
test_labels = tf.boolean_mask(labels, test_mask)
train_labels = tf.boolean_mask(labels, train_mask)

# train and evalate

# define hyper-parameters
output_dim = tf.math.reduce_max(labels)+1

num_epochs = 1000
batch_size = 512
learning_rate = 0.005

keras.utils.set_random_seed(1234)

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Lion(learning_rate)
accuracy_fn = keras.metrics.SparseCategoricalAccuracy(name='acc')
early_stopping = keras.callbacks.EarlyStopping(
	monitor='val_acc',
	patience=100,
	restore_best_weights=True
)

# build model
gat_model = gat.models.GraphAttentionNetworkTransductive(
	features, edges, output_dim.numpy()
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










