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

Our temporal sampler is implemented using C++. Please compile it first with:
```
python setup.py build_ext --inplace
```
## Dataset



## Configuration Files

We provide examples for our configuration files at `./config` directory. Specifying them 
Here is some information about key configurations:

train:

* order: options are 'chorno' or 'random'. 'chorno' stands for training data in chornological order, while 'random' means training data shuffled at the start of every epoch. We only support 'chorno' order for RNN-based node memory , i.e., `memory_type: 'gru'`

scope:

* layer: layer of sampling
* neighbor: neighbors to sample in each layer, starting from inner layer
* strategy: strategy for neighbor sampling, options are 'recent' or 'uniform'

gnn:

* arch: architecture for gnn neighbor aggregator, options are 'transformer' or 'mixer'
* time_enc: time encoding, options are 'learnable' or 'fixed'
* memory_type: node memory, options are 'embedding', 'gru' or 'none'


## Training and Testing

Our training script contains training, evaluation and testing. Self-supervised extrapolation link prediction task:

```
python src/train.py --config config/Wikipedia/example.yml \
                    --data WIKI \
                    --gpu 0  
```

* For inductive setting, add flag `--inductive` to train in inductive setting. We mask some of the nodes from test sets as unseen nodes and remove all edges with these unseen nodes from the training set. The ratio of masked nodes is set with `--ind_ratio`
* 

For testing only, load the saved model parameters and run the test script:

```
python src
```

## Synthetic Dataset Generation

To generate synthetic datasets, run script:

```bash
bash gen_syn_data.sh ${ALPHA} ${BETA}
```


