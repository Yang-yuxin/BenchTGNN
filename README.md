# Towards Ideal Temporal Graph Neural Networks: Evaluations and Conclusions after 10,000 GPU Hours

## Overview

This is the repository for our work **Towards Ideal Temporal Graph Neural Networks: Evaluations and Conclusions after 10,000 GPU Hours**.

## Dependencies

* dgl >= 2.1.0
* numpy >= 1.26.4
* pandas >= 2.2.2
* scikit_learn >= 1.4.2
* setuptools >= 68.0.0
* tensorboard >= 2.15.1
* torch >= 2.3.0
* torch_geometric >= 2.5.3
* tqdm >= 4.65.0
* CUDA >= v10.1
* pybind11 >= 2.6.2
* g++ >= 7.5.0
* openmp >= 201511

Our temporal sampler is implemented using C++. Please compile it first with:
```
python setup.py build_ext --inplace
```

## Dataset

### Format

1. `edges.csv`: The file that stores temporal edge informations. The csv should have the following columns with the header as `,src,dst,time,ext_roll` where each of the column refers to edge index (start with zero), source node index (start with zero), destination node index, time stamp, extrapolation roll (0 for training edges, 1 for validation edges, 2 for test edges). The CSV should be sorted by time ascendingly.
2. `ext_full.npz`: The T-CSR representation of the temporal graph. We provide a script to generate this file from `edges.csv`. You can use the following command to use the script 
    >python gen_graph.py --data \<NameOfYourDataset>
3. `edge_features.pt` (optional): The torch tensor that stores the edge featrues row-wise with shape (num edges, dim edge features). *Note: at least one of `edge_features.pt` or `node_features.pt` should present.*
4. `node_features.pt` (optional): The torch tensor that stores the node featrues row-wise with shape (num nodes, dim node features). *Note: at least one of `edge_features.pt` or `node_features.pt` should present.*


### Real-world datasets
We provide a script to download four of the datasets to `DATA/` (by default), *Wikipedia*, *REDDIT*, *LASTFM* and *MOOC*:
```bash
bash download.sh
```
*UCI*, *College*, and *Flights* can be downloaded [here](https://github.com/qianghuangwhu/benchtemp?tab=readme-ov-file).
Preprocess all the seven datasets used in our work:

```bash
bash preprocess.sh
```

### Synthetic Dataset Generation

To generate synthetic datasets using our method, run script:

```bash
bash gen_syn_data.sh ${ALPHA} ${BETA}
```

Datasets will be generated as `DATA/syn_${ALPHA}_${BETA}`, with the same format as real-world datasets. Currently we don't support node and edge feature generation.

## Configuration Files

We provide examples for our configuration files at `./config` directory. Specifying them 
Here is some information about key configurations:

train:

* order: options are 'chrono' or 'random'. 'chrono' stands for training data in chronological order, while 'random' means training data shuffled at the start of every epoch. We only support 'chrono' order for RNN-based node memory , i.e., `memory_type: 'gru'`

scope:

* layer: layer of sampling
* neighbor: neighbors to sample in each layer, starting from inner layer
* strategy: strategy for neighbor sampling, options are 'recent' or 'uniform', standing for most recent sampling and uniform random sampling

gnn:

* arch: architecture for gnn neighbor aggregator, options are 'transformer' or 'mixer'
* time_enc: time encoding, options are 'learnable' or 'fixed'
* memory_type: node memory, options are 'embedding', 'gru' or 'none'


## Training and Testing

Our training script contains training, evaluation and testing. Self-supervised extrapolation link prediction task:

```bash
python src/train.py --config config/Wikipedia/example.yml \
                    --data WIKI \
                    --gpu ${GPU}
```

* For inductive setting, add flag `--inductive` to train in inductive setting. We mask some of the nodes from test sets as unseen nodes and remove all edges with these unseen nodes from the training set. The ratio of masked nodes is set with `--ind_ratio`




