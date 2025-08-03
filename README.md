# **GCL-ALG: graph contrastive learning with adaptive learnable view generators**

Source code for paper  **GCL-ALG: graph contrastive learning with adaptive learnable view generators**, PeerJ Computer Science, 2025.

# Dependencies

```
python 3.9
pytorch 1.13.1 
pytorch_geometric 2.3.0
```

# Dataset Preparation

```shell
$ python download_dataset.py
```

# Usage

## Semi-supervised Learning

```shell
$ python main.py --exp=joint_cl_exp --semi_split=10 --dataset=IMDB-BINARY --save=joint_cl_exp --epochs=100 --batch_size=128 --lr=0.001
```

## Unsupervised Learning

```
$ python train.py --dataset=IMDB-BINARY --epochs=100 --batch_size=128 --lr=0.001
```

## Transfer Learning

Prepare the Dataset

```shell
$ cd transfer
$ wget http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip
$ unzip chem_dataset.zip
$ rm -rf dataset/*/processed
```

Run the Fine-tuning Experiments

```shell
$ python finetune.py
```

