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
- `betweenness.py`: inductive node regression task for betwenness centrality prediction. This task is done to test GAT ability to learn a node centrality measure. See `closeness.py`.
- `citeseer.py`: transductive node classification task using CiteSeer dataset (Giles et al., 1998). The CiteSeer dataset consists of 3312 scientific publications classified into one of six classes. The citation network consists of 4732 links. Each publication in the dataset is described by a 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary. The dictionary consists of 3703 unique words.
- `closeness.py`: inductive node regression task for closeness centrality prediction. This task is done to test GAT ability to learn a node centrality measure. I used Laplacian eigenvectors decomposition as positional encoding for each node (see Dwivedi et al., 2022). Laplacian eigenvectors associated to the smallest non-trivial eigenvalues are used as input features for each node and inside attention scores as a kind of relative positional encoding.
- `cora.py`: transductive node classification task using Cora dataset (McCallum et al., 2000). The Cora dataset consists of 2708 scientific publications classified into one of seven classes. The citation network consists of 5429 links. Each publication in the dataset is described by a 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary. The dictionary consists of 1433 unique words.
- `optimizers.py`: includes a simple implementation of the Adan optimizer (Xie et al., 2022), a variant of the Adam optimizer (Kingma and Ba, 2014) using a corrected Nesterov momentum (cfr. Nesterov, 1983). For some tasks, this optimizer gives better convergence or generalization than Adam.
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
conda install networkx
```

## Explanation

A graph attention network is characterized by the use of a so-called graph attention layer, which is a function of node features ![\mathbf{X} = \{\mathbf{x}_1,\dots,\mathbf{x}_N\}, \quad \mathbf{x}_i \in \mathbb{R}^F](https://quicklatex.com/cache3/e2/ql_3ea2f4c358cfceb055d3ee710cfa6de2_l3.png), where $N$ is the number of nodes in the graph and $F$ is the number of features in each node. After applying a linear transformation by a weight matrix ![W \in \mathbb{R}^{F'\times F}](https://quicklatex.com/cache3/23/ql_4888e3123a8b4858bca24ef58b63fd23_l3.png) to each node, where $F'$ is the number of output features, we define the attention mechanism ![a : \mathbb{R}^{F'} \times \mathbb{R}^{F'} \rightarrow \mathbb{R}](https://quicklatex.com/cache3/29/ql_f1012565442e3839c953cfb68c477929_l3.png) which computes the attention coefficients

<p align="center">
<img alt="e_{ij} = a(W\mathbf{x}_i, W\mathbf{x}_j)" src="https://quicklatex.com/cache3/4c/ql_022dcc16d1098cf160255ad2619c594c_l3.png">
</p>

that indicates the importance of node $j$'s features to node $i$. We inject graph structure into the mechanism by performing masked attention, meaning that $e_{ij}$ is computed only for nodes ![e_{ij} = a(W\mathbf{x}_i, W\mathbf{x}_j)](https://quicklatex.com/cache3/9e/ql_6183274cb113a42b3ade0fc1eeda629e_l3.png), where ![\mathcal{N}_i](https://quicklatex.com/cache3/9a/ql_38a8afd0d8c58bce466e682d0dba089a_l3.png) is some neighborhood of node $i$ in the graph. In practice, first-order neighborhood is chosen. Attention coefficients are then normalized across $j$ using the softmax function:

<p align="center">
<img alt="\alpha_{ij} = \mathrm{softmax}_j (e_{ij}) = \frac{\mathrm{exp}(e_{ij})}{\sum_{k \in \mathcal{N}_i} \mathrm{exp}(e_{ij})}." src="https://quicklatex.com/cache3/c3/ql_3cd5303047919d04b69f55bf94dcb1c3_l3.png">
</p>

Velickovic et al. (2018) suggest to use a single-layer feedforward neural network for the attention mechanism, parametrized by a weight vector ![\mathbf{a} \in \mathbb{R}^{2F'}](https://quicklatex.com/cache3/00/ql_2e11c75cd62f4da57884eb00cde74700_l3.png) and applying the Leaky ReLU nonlinearity (with negative slope $\alpha = 0.2$, see Redmon et al., 2015), i.e.:

<p align="center">
<img alt="e_{ij} = \mathrm{LeakyReLU}(\mathbf{a}^T [W\mathbf{h}_i || W\mathbf{h}_j])}," src="https://quicklatex.com/cache3/28/ql_85b3fc4b38f36d3f92eff8b84c595228_l3.png">
</p>

where $\cdot^T$ represent transposition and $||$ is the concatenation operation. The normalized attention coefficient are then used to compute a linear combination of the features corresponding to them, with a possible nonlinearity $\sigma$ at the end:

<p align="center">
<img alt="\mathbf{h}_i = \sigma \left(\sum_{j \in \mathcal{N}_i} \alpha_{ij} W \mathbf{x}_{j} \right)." src="https://quicklatex.com/cache3/1e/ql_c685104789ec2a0db31c7b7b2b006c1e_l3.png">
</p>

To stabilize training, it is possible to employ multi-head attention, based on the work by Vaswani et al. (2017). If $K$ is the number of attention heads, then we execute an attention mechanism for each head and concatenate the output features:

<p align="center">
<img alt="\DeclareMathOperator*{\bigVert}{\big\Vert} \mathbf{h}_i = \bigVert_{k=1}^K \sigma \left(\sum_{j \in \mathcal{N}_i} \alpha_{ij}^k W^k \mathbf{x}_{j} \right)." src="https://quicklatex.com/cache3/25/ql_e1c72175a14b66ebf3d982c478ce0f25_l3.png">
</p>

Note that, in this case, the output will consist of $KF'$ features for each node. For the last layer, rather than concatenation, averaging is usually preferred:

<p align="center">
<img alt="\mathbf{h}_i = \sigma \left(\frac{1}{K} \sum_{k=1}^K \sum_{j \in \mathcal{N}_i} \alpha_{ij}^k W^k \mathbf{x}_{j} \right)." src="https://quicklatex.com/cache3/89/ql_9c0b748f479eeada02bc246839f4a289_l3.png">
</p>

## Bibliography

- Y. Nesterov, _A method of solving a convex programming problem with convergence rate O(1/k^2)_, Dokl. Akad. Nauk SSSR, Vol. 269, No. 3, 543-547, 1983.
- C. L. Giles, K. D. Bollacker, S. Lawrence, _CiteSeer: An Automatic Citation Indexing System_, Proceedings of the third ACM conference on Digital libraries, 1998.
- A. K. McCallum, K. Nigam, J. Rennie, K. Seymore, _Automating the Construction of Internet Portals with Machine Learning_, Information Retrieval Journal 3(2): 127-163, 1999.
- P. Sen, G. Namata, M. Bilgic, L. Getoor, B. Gallagher, T. Eliassi-Rad, _Collective Classification in Network Data_, AI Magazine, 29(3), 93, 2008.
- X. Glorot, Y. Bengio, _Understanding the difficulty of training deep feedforward neural networks_, International Conference on Artificial Intelligence and Statistics, 2010.
- N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, R. Salakhutdinov, _Dropout: A Simple Way to Prevent Neural Networks from Overfitting_, Journal of machine learning research, 2014.
- D. P. Kingma, J. L. Ba, _Adam: A Method for Stochastic Optimization_, 3rd International Conference for Learning Representations, San Diego, 2015.
- C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, Z. Wojna, _Rethinking the Inception Architecture for Computer Vision_, Computer Vision and Pattern Recognition, 2015.
- J. L. Ba, J. R. Kiros, G. E. Hinton, _Layer Normalization_, Advances in NIPS, Deep Learning Symposium, 2016.
- W. L. Hamilton, R. Ying, J. Leskovec, _Inductive Representation Learning on Large Graphs_, Advances in Neural Information Processing Systems 30, 2017.
- A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, I. Polosukhin, _Attention Is All You Need_, Neural Information Processing Systems, 2017.
- P. Velickovic, G. Cucurull, A. Casanova, A. Romero, P. Liò, Y. Bengio, _Graph Attention Networks_, International Conference on Learning Representations, 2017.
- S. Merity, _Single Headed Attention RNN: Stop Thinking With Your Head_, 2019 (not peer-reviewed).
- S. Brody, U. Alon, E. Yahav, _How Attentive are Graph Attention Networks?_, ICLR 2022.
- V. P. Dwivedi, C. K. Joshi, A. T. Luu, T. Laurent, Y. Bengio, X. Bresson, _Benchmaking Graph Neural Networks_, JMLR 2022.
- X. Xie, P. Zhou, H. Li, Z. Lin, S. Yan, _Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models_, IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022.
