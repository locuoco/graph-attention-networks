# graph-attention-networks
Graph Attention Networks (GATs) for node classification in transductive and inductive tasks. This implementation refers to the [article by Velickovic et al. (2018)](https://arxiv.org/pdf/1710.10903) and it is based on the implementation of `GATConv` from [Spektral library](https://github.com/danielegrattarola/spektral). This code uses Tensorflow/Keras frameworks and has the following dependencies:

- Keras 3.0+ (backend-agnostic, tested with Tensorflow and JAX)
- Deep Graph Library (for graphs dataset loading)

The multi-headed graph attention layer has been employed in two different tasks. For transductive learning, the nodes features of one graph are used to calculate all the outputs for each node. After the forward pass, all non-relevant nodes are masked out, so that it's possible to train the neural network based on the chosen training nodes. The transductive model is then evaluated on validation and test nodes. For inductive learning, entire graphs are used for training, while completely different graphs, not seen during training, are used for validation and test. The GAT architecture has been shown to be especially effective with this latter kind of task, reaching an F1 micro-averaged score of 97.5%.
