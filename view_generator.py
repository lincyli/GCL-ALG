from torch_geometric.nn import VGAE
import torch,gc
import torch.nn as nn
from torch.nn import functional as F, DataParallel
import copy
from torch_geometric.utils import subgraph, to_networkx, add_self_loops
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool
import math
import networkx as nx
from GinSample import GINConv_weight

class GIN_Classifier(torch.nn.Module):
    def __init__(self, dataset, dim):
        super().__init__()

        num_features = dataset.num_features

        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, dataset.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

    def forward_cl(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        return x

class selfAttention(nn.Module):
    def __init__(self, input_size, hiddden_size):
        super(selfAttention, self).__init__()
        self.input_size = input_size
        self.hiddden_size = hiddden_size
        self.key_layer = nn.Linear(input_size, hiddden_size)
        self.query_layer = nn.Linear(input_size, hiddden_size)
        self.value_layer = nn.Linear(input_size, hiddden_size)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        d = x.shape[-1]
        w_k = self.key_layer(x)
        w_q = self.query_layer(x)
        w_v = self.value_layer(x)

        attention_scores = torch.matmul(w_q, w_k.t())
        d_ = math.sqrt(d)
        attention_scores = attention_scores / d_
        attention_probs = F.softmax(attention_scores, dim=-1)

        context = torch.matmul(attention_probs, w_v)
        context = context.view(*x.shape)

        # return context*self.gamma
        return context

class GIN_Node_With_Edge_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim,device):
        super(GIN_Node_With_Edge_Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        nn1 = Sequential(Linear(input_dim, hidden_dim*2), ReLU(), Linear(hidden_dim*2, hidden_dim))
        self.conv1 = GINConv_weight(device,nn1)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim).to(device)
        nn2 = Sequential(Linear(hidden_dim, hidden_dim*2), ReLU(), Linear(hidden_dim*2, 2))
        self.conv2 = GINConv_weight(device,nn2)
        self.bn2 = torch.nn.BatchNorm1d(2).to(device)

    def forward(self, x, edge_index, edge_attr, edge_weight):
        x = F.relu(self.conv1(x, edge_index,edge_attr, edge_weight))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index, edge_attr,edge_weight))
        x = self.bn2(x)
        return x


class mlp_edge_encoder(torch.nn.Module):
    def __init__(self, mlp_dim,weight,add_mask=False):
        super(mlp_edge_encoder, self).__init__()
        self.mlp_dim = mlp_dim
        self.add_mask = add_mask
        self.weight = weight
        if self.add_mask == True:
            self.input_dim = 3 * 2
        else:
            self.input_dim = 2 * 2

        if self.weight == True:
            out_dim = 1
        else:
            out_dim = 2

        self.mlp_edge_model = Sequential(
            Linear(self.input_dim, self.mlp_dim),
            ReLU(),
            Linear(self.mlp_dim, out_dim)
        )

    def forward(self, edge_emb):
        edge_logit = self.mlp_edge_model(edge_emb)
        return edge_logit

class GIN_NodeWeightEncoder(torch.nn.Module):
    def __init__(self, dataset, dim, add_mask=False):
        super().__init__()
        self.dim = dim
        num_features = dataset.num_features
        nn1 = Sequential(Linear(num_features, dim*2), ReLU(), Linear(dim*2, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)
        self.attn1 = selfAttention(dim, dim)
        if add_mask == True:
            nn2 = Sequential(Linear(dim, dim*2), ReLU(), Linear(dim*2, 3))
            self.conv2 = GINConv(nn2)
            self.bn2 = torch.nn.BatchNorm1d(3)
            self.attn2 = selfAttention(3, 3)

        else:
            nn2 = Sequential(Linear(dim, dim*2), ReLU(), Linear(dim*2, 2))
            self.conv2 = GINConv(nn2)
            self.bn2 = torch.nn.BatchNorm1d(2)

            self.attn2 = selfAttention(2, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = self.attn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = self.attn2(x)
        return x



class ViewGenerator(VGAE):
    def __init__(self, dataset, dim, mlp_dim, encoder, mlp_edge_encoder, add_mask=False):
        self.add_mask = add_mask
        self.dim = dim
        encoder = encoder(dataset, dim, self.add_mask)
        super().__init__(encoder=encoder)
        self.edge_encoder1 = mlp_edge_encoder(mlp_dim, False,add_mask)
        self.edge_encoder2 = mlp_edge_encoder(mlp_dim, True,add_mask)


    def forward(self, data_in, requires_grad):
        data = copy.deepcopy(data_in)
        x, edge_index = data.x, data.edge_index
        edge_attr = None
        G = to_networkx(data)
        if data.edge_attr is not None:
            edge_attr = data.edge_attr
        data.x = data.x.float()

        x = x.float()
        x.requires_grad = requires_grad

        node_emb = self.encoder(data)

        src, dst = edge_index[0], edge_index[1]
        emb_src = node_emb[src]
        emb_dst = node_emb[dst]
        edge_emb = torch.cat([emb_src, emb_dst], dim=1)

        edge_logits = self.edge_encoder2(edge_emb)
        temperature = 1.0
        bias = 0.0+0.0001
        eps = (bias-(1-bias))*torch.rand(edge_logits.size())+(1-bias)
        gate_input = torch.log(eps)-torch.log(1-eps)
        gate_input = gate_input.to(edge_index.device)
        gate_input = (gate_input+edge_logits) / temperature
        edge_weight = torch.sigmoid(gate_input).squeeze()


        edge_emb1 = self.edge_encoder1(edge_emb)
        edge_measure_result = edge_degree_eval(G,edge_index,data.num_nodes)
        edge_emb = edge_emb1 * edge_measure_result.view(-1,1)


        node_measure_result = degree_centrality_eval(G, data.num_nodes, x.device)
        Gin_encoder = GIN_Node_With_Edge_Encoder(node_emb.shape[1], self.dim,edge_index.device)
        x_emb = Gin_encoder(node_emb, edge_index, edge_attr, edge_weight)
        node_emb = x_emb * node_measure_result.view(-1, 1)

        edge_sample = F.gumbel_softmax(edge_emb, hard=True)
        real_edge_sample = edge_sample[:, 0]
        keep_edge_idx = torch.nonzero(real_edge_sample, as_tuple=False).view(-1, )
        keep_edge_src = edge_index[0][keep_edge_idx].unsqueeze(dim=1)
        keep_edge_dst = edge_index[1][keep_edge_idx].unsqueeze(dim=1)
        edge_index = torch.concat([keep_edge_src, keep_edge_dst], dim=1).t()  # 12557
        if edge_attr is not None:
            edge_attr = edge_attr[keep_edge_idx]

        sample = F.gumbel_softmax(node_emb, hard=True)
        real_sample = sample[:, 0]
        attr_mask_sample = None
        if self.add_mask == True:
            attr_mask_sample = sample[:, 2]
            keep_sample = real_sample + attr_mask_sample
        else:
            keep_sample = real_sample
        keep_idx = torch.nonzero(keep_sample, as_tuple=False).view(-1, )

        edge_index, edge_attr = subgraph(keep_idx, edge_index, edge_attr, num_nodes=data.num_nodes)
        x = x * keep_sample.view(-1, 1)

        if self.add_mask == True:
            attr_mask_idx = attr_mask_sample.bool()
            token = data.x.detach().mean()
            x[attr_mask_idx] = token
        data.x = x
        data.edge_index = edge_index
        if data.edge_attr is not None:
            data.edge_attr = edge_attr
        return keep_sample, data


def edge_degree_eval(G,edge_index,num_nodes):
    src = edge_index[0]
    dst = edge_index[1]
    degree = nx.degree(G)
    x = [degree[i] for i in range(num_nodes)]
    edge_cents = [(x[src[i]]+x[dst[i]])/2.0 for i in range(len(edge_index[0]))]

    return torch.tensor(edge_cents, dtype=torch.float32).to(edge_index.device)

def degree_centrality_eval(G,num_nodes,device):
    x = nx.degree_centrality(G)
    x = [x[i] for i in range(num_nodes)]
    return torch.tensor(x, dtype=torch.float32).to(device)










