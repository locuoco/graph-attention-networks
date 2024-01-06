import tensorflow as tf
import dgl

dataset = dgl.data.CoraGraphDataset()
graph = dataset[0]

# get split masks
val_mask = graph.ndata['val_mask']
test_mask = graph.ndata['test_mask']
# (training mask was not set properly in graph.ndata['train_mask'])
train_mask = tf.math.logical_xor(tf.constant([True]), (val_mask | test_mask))

# get node features
features = graph.ndata['feat']

# get ground-truth labels
labels = graph.ndata['label']

