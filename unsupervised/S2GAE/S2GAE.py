import torch
import torch.nn as nn
from torch.nn import functional as F
from unsupervised.S2GAE.S2GAEConv import SAGEConv

class SAGE_mgaev2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE_mgaev2, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def mask_decode(self, x):
        x = torch.cat([self.n_emb.weight, x], dim=-1)
        for lin in self.mask_lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.mask_lins[-1](x)
        return x

    def forward(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(F.relu(x))
        return xx

    def outEmb(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(x)
        x = torch.cat(xx, dim=1)
        return x