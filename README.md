# Graph Attention Networks
Graph Attention Networks (GATs) for node classification in transductive and inductive tasks. This implementation refers to the [article by Velickovic et al. (2018)](https://arxiv.org/pdf/1710.10903) and it is based on the implementation of `GATConv` from [Spektral library](https://github.com/danielegrattarola/spektral). A correction to this neural network architecture has been suggested by [Brody et al. (2021)](https://arxiv.org/pdf/2105.14491): this modification is referred to as GATv2. This code uses Keras framework and has the following dependencies:

- Keras 3.0+ (backend-agnostic, tested with Tensorflow and JAX)
- Deep Graph Library (DGL, for graphs dataset loading and calculation of Laplacian eigenvectors)
- NetworkX (for calculation of centrality measures)

Keras 3.0+ is not available on GPU on Windows. I suggest using WSL to enable GPU acceleration for Windows users. Note further that, while the code is mostly backend-agnostic, incompatible backends for Keras and DGL may cause runtime errors during execution. On Linux/WSL, you may choose which backend to use for Keras and DGL by editing `~/.keras/keras.json` and `~/.dgl/config.json` files, respectively. Tensorflow support has been deprecated on DGL latest versions, but it still works fine as of DGL 2.3.0.

The multi-headed graph attention layer has been employed in two different tasks. For transductive learning, the node features of one graph are used to calculate all the outputs for each node. All non-relevant nodes are masked out, so that it's possible to train the neural network based on the chosen training nodes. The transductive model is then evaluated on validation and test nodes. For inductive learning, entire graphs are used for training, while completely different graphs, not seen during training, are used for validation and test. The GAT architecture has been shown to be especially effective with this latter kind of task, reaching an F1 micro-averaged test score of 99.43% for protein-protein interaction (PPI) dataset. The model uses regularization techniques such as dropout (Szegedy et al., 2015) and layer normalization (Ba et al., 2016), along with a feedforward layer after each GAT layer, inspired by Transformer architecture (Vaswani et al., 2017) and by the "Boom" layer of SHA-RNN (Merity, 2019).

## Repository structure

- `data`: directory containing pickle files (cached data objects to avoid recomputation). If a script fails to load a pickle file (maybe due to incompatible module versions used for saving), try to remove it to force recomputation.
- `gat`: directory containing GAT layer and models.
  - `layers.py`: it contains the `MultiHeadGraphAttention` layer, implementing the original GAT (Velickovic et al., 2018) and its variant GATv2 (Brody et al., 2021).
  - `models.py`: it contains two specialized transductive models `GraphAttentionNetworkTransductive` (used for Cora and Citeseer datasets) and `GraphAttentionNetworkTransductive2` (used for Pubmed dataset) and a general customizable inductive model `GraphAttentionNetworkInductive` (used for PPI dataset and closeness centrality task). The two transductive models are currently built specifically for the specified tasks, so they may need to be modified slightly for new ones.
- `weights`: directory which will be populated with networks parameters weights for PPI and closeness centrality models after training.
- `citeseer.py`: transductive node classification task using CiteSeer dataset (Giles et al., 1998). The CiteSeer dataset consists of 3312 scientific publications classified into one of six classes. The citation network consists of 4732 links. Each publication in the dataset is described by a 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary. The dictionary consists of 3703 unique words.
- `closeness.py`: inductive node regression task for closeness centrality prediction. This task is done to test GAT ability to learn a node centrality measure. I used Laplacian eigenvectors decomposition as positional encoding for each node (see Dwivedi et al., 2022). Laplacian eigenvectors associated to the smallest non-trivial eigenvalues are used as input features for each node and inside attention scores as a kind of relative positional encoding.
- `cora.py`: transductive node classification task using Cora dataset (McCallum et al., 2000). The Cora dataset consists of 2708 scientific publications classified into one of seven classes. The citation network consists of 5429 links. Each publication in the dataset is described by a 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary. The dictionary consists of 1433 unique words.
- `ppi.py`: inductive node classification task using PPI dataset (Hamilton et al., 2017). Each graph in the dataset represents the interactions between proteins in a specific human tissue. Nodes represent proteins and edges represent interactions between them. The average graph in the dataset contains around 2373 nodes (i.e., proteins), and the average degree (the number of interactions per protein) is 28.8. Positional gene sets, motif gene sets, and immunological signatures are used as features, while 121 gene ontology sets are used as labels.
- `pubmed.py`: transductive node classification task using PubMed dataset (Sen et al., 2008). The PubMed dataset consists of 19717 scientific publications from PubMed database pertaining to diabetes classified into one of three classes. The citation network consists of 44338 links. Each publication in the dataset is described by a term frequency/inverse document frequency weighted word vector from a dictionary which consists of 500 unique words.

## Dependencies installation

Under Windows, it's possible to install Keras 3.0+ with GPU acceleration using WSL. I recommend following [this guide](https://medium.com/@pratik_davidson/install-tensorflow-2-16-and-keras-3-with-gpu-acceleration-on-windows-wsl2-a6bc2a7d77cb) for a clean installation using Tensorflow backend. You can also take a look at [this](https://keras.io/getting_started/) for other backends.

For DGL installation, you can use (using Anaconda):
```sh
conda install -c conda-forge 'dgl>=2.3'
```
Note further that, while the code is mostly backend-agnostic, incompatible backends for Keras and DGL may cause runtime errors during execution. On Linux/WSL, you may choose which backend to use for Keras and DGL by editing `~/.keras/keras.json` and `~/.dgl/config.json` files, respectively. Tensorflow support has been deprecated on DGL latest versions, but it still works fine as of DGL 2.3.0 (current code written for Tensorflow is unlikely to be removed, although bugs and problems will not be corrected).

For NetworkX installation, you can use (using Anaconda):
```sh
conda install networkx'
```

## Explanation
