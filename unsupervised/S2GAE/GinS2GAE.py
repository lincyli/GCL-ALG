import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool


class GIN_mgaev2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, decoder_mask='nmask', eps: float = 0.,  bias=True, xavier=True):
        super(GIN_mgaev2, self).__init__()
        self.decoder_mask = decoder_mask
        self.initial_eps = eps
        self.convs = torch.nn.ModuleList()
        self.act = torch.nn.ReLU()
        for i in range(num_layers - 1):
            start_dim = hidden_channels if i else in_channels
            nn = Sequential(Linear(start_dim, hidden_channels, bias=bias),
                            self.act,
                            Linear(hidden_channels, hidden_channels, bias=bias))
            # if xavier:
            #     self.weights_init(nn)
            conv = GINConv(nn)
            self.convs.append(conv)
        nn = Sequential(Linear(hidden_channels, hidden_channels, bias=bias),
                        self.act,
                        Linear(hidden_channels, hidden_channels, bias=bias))
        # if xavier:
        #     self.weights_init(nn)
        conv = GINConv(nn)
        self.convs.append(conv)

        self.dropout = dropout

    def weights_init(self, module):
        for m in module.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def reset_parameters(self):
        for conv in self.convs:
            # self.weights_init(conv.nn)
            # conv.eps.data.fill_(self.initial_eps)
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
