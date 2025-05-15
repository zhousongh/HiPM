import dgl
import torch
from typing import List
import torch.nn as nn
import dgl.nn.pytorch as dglnn


class DGLMPNNLayer(nn.Module):
    def __init__(self,
                    hid_dim,
                    edge_func,
                    residual):
        super(DGLMPNNLayer, self).__init__()
        self.hidden_dim = hid_dim
        self.node_conv = dglnn.NNConv(hid_dim, hid_dim, edge_func, 'sum', residual)

    def forward(self, g, nf, initial_ef):
        unm = self.node_conv(g, nf, initial_ef)
        return unm

class DGLMessagePassingNetwork(nn.Module):
    """A flexible message passing neural network with adjustable layers."""
    def __init__(self,
                 hid_dim,
                 edge_func,
                 num_layers,
                 residual=True):
        super(DGLMessagePassingNetwork, self).__init__()
        self.hidden_dim = hid_dim
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(DGLMPNNLayer(self.hidden_dim, edge_func, residual))

    def forward(self, g, node_features, initial_edge_features):
        for layer in self.layers:
            node_features = layer(g, node_features, initial_edge_features)
        return node_features
   
       
class MLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 dropout_prob: float,
                 num_neurons: List[int] = [150],
                 hidden_act: str = 'LeakyReLU',
                 out_act: str = 'Identity',
                 input_norm: str = 'layer',
                 ):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob
        self.num_neurons = num_neurons
        self.hidden_act = getattr(nn, hidden_act)()
        self.out_act = getattr(nn, out_act)()
        input_dims = [input_dim, num_neurons]
        output_dims = [num_neurons, output_dim]
        self.act_func = nn.Sigmoid()
        self.layers = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(input_dims, output_dims)):
            is_last = True if i == len(input_dims) - 1 else False
            self.layers.append(nn.Linear(in_dim, out_dim))
            if is_last:
                self.layers.append(self.out_act)
            else:
                self.layers.append(self.hidden_act)

        if dropout_prob > 0.0:
            self.dropout = nn.Dropout(dropout_prob)

        if input_norm is not None:
            assert input_norm in ['batch', 'layer']

            if input_norm == 'batch':
                self.input_norm = nn.BatchNorm1d(input_dim)
            if input_norm == 'layer':
                self.input_norm = nn.LayerNorm(input_dim)

    def forward(self, xs):
        if hasattr(self, 'input_norm'):
            xs = self.input_norm(xs)
        for i, layer in enumerate(self.layers):
            if i != 0 and hasattr(self, 'dropout'):
                xs = self.dropout(xs)
            xs = layer(xs)

        return xs


class Readout(nn.Module):
    def __init__(self, args_dict, ntype: str, use_attention: bool):
        super(Readout, self).__init__()
        self.ntype = ntype
        self.use_attention = use_attention
        self.linear = nn.Linear(args_dict['hid_dim'], 1)

    def forward(self, g, nf):
        if self.use_attention:
            g.nodes[self.ntype].data['nw'] = self.linear(nf)
            weights = dgl.softmax_nodes(g, 'nw', ntype=self.ntype)
            with g.local_scope():
                g.nodes[self.ntype].data['w'] = weights
                g.nodes[self.ntype].data['feat'] = nf
                weighted_mean_rd = dgl.readout_nodes(
                    g, 'feat', 'w', op='sum', ntype=self.ntype)
                max_rd = dgl.readout_nodes(
                    g, 'feat', op='max', ntype=self.ntype)
                return torch.cat([weighted_mean_rd, max_rd], dim=1)
        else:
            with g.local_scope():
                g.nodes[self.ntype].data['feat'] = nf
                mean_rd = dgl.readout_nodes(
                    g, 'feat', op='mean', ntype=self.ntype)
                max_rd = dgl.readout_nodes(
                    g, 'feat', op='max', ntype=self.ntype)
                return torch.cat([mean_rd, max_rd], dim=1)
            

class LinearBn(nn.Module):
    def __init__(self, in_channel, out_channel, act=None):
        super(LinearBn, self).__init__()
        self.linear = nn.Linear(in_channel, out_channel)
        self.bn = nn.BatchNorm1d(out_channel)
        self.act = act

    def forward(self, x):
        x = self.linear(x)
        size = x.size()
        x = x.view(-1, x.size()[-1], 1)
        x = self.bn(x)
        x = x.view(size)
        if self.act is not None:
            x = self.act(x)
        return x

