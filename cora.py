import tensorflow as tf
from tensorflow import keras
import dgl

import gat

dataset = dgl.data.CoraGraphDataset()
graph = dataset[0]
nodes_indices = graph.nodes()
edges = tf.transpose(tf.convert_to_tensor(graph.edges()))

print('Edges shape:\t\t', edges.shape)
print(edges)

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
hidden_units = 200
num_heads = 4
num_layers = 3
output_dim = tf.math.reduce_max(labels)+1

num_epochs = 100
batch_size = 256
learning_rate = 0.3
momentum = 0.9

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.SGD(learning_rate, momentum=momentum)
accuracy_fn = keras.metrics.SparseCategoricalAccuracy(name='acc')
early_stopping = keras.callbacks.EarlyStopping(
	monitor='val_acc',
	min_delta=1e-5,
	patience=5,
	restore_best_weights=True
)

# build model
gat_model = gat.GraphAttentionNetwork(
	features, edges, hidden_units, num_heads, num_layers, output_dim
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










