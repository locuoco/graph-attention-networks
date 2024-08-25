# graph-attention-networks
Graph Attention Networks (GATs) for node classification in transductive and inductive tasks. This implementation refers to the [article by Velickovic et al. (2018)](https://arxiv.org/pdf/1710.10903) and it is based on the implementation of `GATConv` from [Spektral library](https://github.com/danielegrattarola/spektral). This code uses Tensorflow/Keras frameworks and has the following dependencies:

- Tensorflow 2.6.0 / Keras 2.6.0
- Tensorflow addons 0.13.0 (discontinued support as of 2024)
- Deep Graph Library 1.1.1 (with CUDA 11.3 support)

Note that installing tensorflow with GPU support will require installation of the Python package `tensorflow-gpu`. Tensorflow addons is used for `tfa.metrics.F1Score`. If Tensorflow 2.16+ / Keras 3 is supported, this class can be found in either `tf.keras.metrics.F1Score` or `keras.metrics.F1Score`.

The multi-headed graph attention layer has been employed in two different tasks. For transductive learning, the nodes features of one graph are used to calculate all the outputs for each node. After the forward pass, all non-relevant nodes are masked out, so that it's possible to train the neural network based on the chosen training nodes. The transuctive model is then evaluated on validation and test nodes. For inductive learning, entire graphs are used for training and different graphs, not seen during training, are used for validation and test. The GAT architecture has been shown to be especially effective with this latter kind of task, reaching an F1 micro-averaged score of 97.3%.
