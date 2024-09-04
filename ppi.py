import os

# Set environment variables for JAX memory limits
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'

import keras
import dgl

import gat.models

load_last_weights = False
continue_training = False
initial_epoch = 0

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
		edges = keras.ops.transpose(keras.ops.convert_to_tensor(graph.edges(), dtype='int32'))

		# get node features
		features = graph.ndata['feat']

		# get ground-truth labels
		mode_labels[i].append(graph.ndata['label'])

		mode_graphs[i].append((features, edges))

# train and evaluate

# define hyper-parameters
output_dim = int(keras.ops.shape(train_labels[0])[-1])

num_epochs = 10000
#batch_size = 1 # number of graphs per batch
learning_rate = 0.001

keras.utils.set_random_seed(1234)
random_gen = keras.random.SeedGenerator(1234)

loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam(learning_rate)
accuracy_fn = keras.metrics.BinaryAccuracy(name='acc')
f1_fn = keras.metrics.F1Score(average='micro', threshold=0.5, name='f1_score')
early_stopping = keras.callbacks.EarlyStopping(
	monitor='val_f1_score',
	patience=200,
	mode='max',
	restore_best_weights=True
)

# build model
gat_model = gat.models.GraphAttentionNetworkInductive(output_dim, random_gen)

# compile model
gat_model.compile(loss=loss_fn, optimizer=optimizer, metrics=[accuracy_fn, f1_fn])

weightsfile = './weights/ppi.weights.h5'

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

test_loss, test_accuracy, test_f1 = gat_model.evaluate(test_generator, verbose=0)

gat_model.save_weights(weightsfile)

print('--'*38 + f'\nTest loss: {test_loss:.4f}, accuracy: {test_accuracy*100:.3f}%, F1 score: {test_f1*100:.3f}%')










