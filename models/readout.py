import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import MultiheadAttention


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
                return torch.cat([weighted_mean_rd, max_rd], dim=1),weights
        else:
            with g.local_scope():
                g.nodes[self.ntype].data['feat'] = nf
                mean_rd = dgl.readout_nodes(
                    g, 'feat', op='mean', ntype=self.ntype)
                max_rd = dgl.readout_nodes(
                    g, 'feat', op='max', ntype=self.ntype)
                return torch.cat([mean_rd, max_rd], dim=1)
