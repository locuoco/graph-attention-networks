import os
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
import dgl

import gat.models

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
		# get edges
		edges = tf.transpose(tf.convert_to_tensor(graph.edges()))

		# get node features
		features = graph.ndata['feat']

		# get ground-truth labels
		mode_labels[i].append(graph.ndata['label'])

		mode_graphs[i].append([features, edges])

# train and evalate

# define hyper-parameters
output_dim = tf.shape(train_labels[0])[-1]

num_epochs = 1000
batch_size = 1 # number of graphs per batch
learning_rate = 0.005

tf.random.set_seed(1234)

loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam(learning_rate)
accuracy_fn = tf.metrics.BinaryAccuracy(name='acc')
f1_fn = tfa.metrics.F1Score(num_classes=output_dim, average='micro', threshold=0.5, name='f1_score')
early_stopping = keras.callbacks.EarlyStopping(
	monitor='val_f1_score',
	min_delta=1e-5,
	patience=100,
	restore_best_weights=True
)

# build model
gat_model = gat.models.GraphAttentionNetworkInductive(output_dim)

# compile model
gat_model.compile(loss=loss_fn, optimizer=optimizer, metrics=[accuracy_fn, f1_fn])

val_generator = gat.models.DataGenerator(val_graphs, val_labels)
test_generator = gat.models.DataGenerator(test_graphs, test_labels)
train_generator = gat.models.DataGenerator(train_graphs, train_labels)

gat_model.fit(
	train_generator,
	validation_data=val_generator,
	epochs=num_epochs,
	callbacks=[early_stopping],
	verbose=2,
)

test_loss, test_accuracy, test_f1 = gat_model.evaluate(test_generator, verbose=0)

gat_model.save('ppi.keras')

print('--'*38 + f'\nTest loss: {test_loss}, accuracy: {test_accuracy*100:.1f}, F1 score: {test_f1*100:.1f}%')










