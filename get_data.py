import os.path as osp
import re

import torch
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.utils import degree
import torch_geometric.transforms as T
from feature_expansion import FeatureExpander
from image_dataset import ImageDataset
# from tu_dataset import TUDatasetExt
from torch_geometric.datasets import TUDataset

from IPython import embed


def get_dataset(name, sparse=True, root=None):
    if root is None or root == '':
        path = osp.join(osp.expanduser('~'), 'pyG_data', name)
    else:
        path = osp.join(root)

    pre_transform = None



    dataset = TUDataset(path, name,
        pre_transform=pre_transform,
        use_node_attr=True)
    dataset = dataset.shuffle()
    # embed()
    # exit()

    dataset.data.edge_attr = None

    return dataset