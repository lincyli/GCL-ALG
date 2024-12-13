import warnings
warnings.filterwarnings("ignore")
from typing import Callable, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch_sparse import SparseTensor


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)
class GINConv_weight(MessagePassing):
    def __init__(self, device,nn: Callable, eps: float = 0., train_eps: bool = False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GINConv_weight, self).__init__(**kwargs)
        self.nn = nn.to(device)
        self.initial_eps = eps

        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))

        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()


    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, edge_weight: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_weight=edge_weight, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps.to(x_r.device)) * x_r.to(x_r.device)

        return self.nn(out)

    def message(self, x_j: Tensor,edge_attr, edge_weight) -> Tensor:
        return F.relu(x_j) if edge_weight is None else F.relu(x_j) * edge_weight.view(-1, 1)

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)