import torch
import torch.nn as nn
import dgl
from .layers import DGLMessagePassingNetwork, MLP, LinearBn, Readout
import torch.nn.init as init
from scipy.cluster.hierarchy import linkage
import copy
import numpy as np
from typing import Dict


class AdaptivePrompter(nn.Module):
    def __init__(self, hidden_dim: int, task_num: int, args_dict: Dict):
        super().__init__()
        self.device = args_dict['device']
        self.hidden_dim = hidden_dim
        self.task_num = task_num
        
        self.prompts = nn.ParameterList([
            nn.Parameter(torch.normal(mean=0, std=0.01, size=(task_num,), dtype=torch.float32, device=self.device))
            for _ in range(task_num)
        ])
        self.Grad_Matrix = torch.zeros(size=(task_num, task_num), device=self.device)
        self.mlp = MLP(
            input_dim=hidden_dim,
            output_dim=task_num,
            dropout_prob=args_dict['drop_rate'],
            num_neurons=args_dict['num_neurons'],
            input_norm=args_dict['input_norm']
        )
        self.mapper = {i: [i] for i in range(task_num)}
        self.P = self.construct_P()
        self.to(self.device)

    def updateGradMatrix(self):
        for task in range(self.task_num):
            self.Grad_Matrix[task] += self.prompts[task].grad.detach()

    def reset_cluster_setting(self):
        self.prompts = self.prompts[:self.task_num]
        self.mapper = {i: [i] for i in range(self.task_num)}
        self.Grad_Matrix = torch.zeros(
            (self.task_num, self.task_num), device=self.device)

    def HierarchicalCluster(self):
        def cosine_distance(u, v):
            return 1 - (u @ v) / (np.linalg.norm(u) * np.linalg.norm(v))

        def dfs(k: int, parent: list):
            if tree[k] == [-1, -1]:
                self.mapper[k] = copy.deepcopy(parent) + [k]
            else:
                parent.append(k)
                dfs(tree[k][0], parent)
                dfs(tree[k][1], parent)
                parent.pop()

        Z = linkage(self.Grad_Matrix.cpu().numpy(),
                    method='average', metric=cosine_distance)

        tree = {task: [-1, -1] for task in range(self.task_num)}
        for item in Z:
            c1, c2 = item[0].astype(int), item[1].astype(int)
            new_prompt = self.prompts[c1] + self.prompts[c2]
            self.prompts.append(new_prompt)

            tree[len(self.prompts)-1] = [c1, c2]

        dfs(len(self.prompts)-1, parent=[])

        self.P = self.construct_P()

    def construct_P(self):
        P = []
        for task in range(self.task_num):
            group = self.mapper[task]
            prompt = torch.stack([self.prompts[i] for i in group]).sum(dim=0)
            P.append(prompt)
        return torch.stack(P)

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.mlp(representation), self.P.T)


class AttnLocalAugmentation(nn.Module):
    '''
    LocalAugmentation module from HimGNN(https://academic.oup.com/bib/article/24/5/bbad305/7245716?login=false)
    '''
    def __init__(self, args_dict):
        super(AttnLocalAugmentation, self).__init__()
        self.linear_layers = nn.ModuleList(
            [nn.Linear(args_dict['hid_dim'], args_dict['hid_dim'], bias=False) for _ in range(3)])
        self.W_o = nn.Linear(args_dict['hid_dim'], args_dict['hid_dim'])
        self.heads = args_dict['heads']
        self.d_k = args_dict['hid_dim'] // args_dict['heads']

    def forward(self, fine_messages, coarse_messages, motif_features):
        batch_size = fine_messages.shape[0]
        hid_dim = fine_messages.shape[-1]
        Q = motif_features
        K = []
        K.append(fine_messages.unsqueeze(1))
        K.append(coarse_messages.unsqueeze(1))
        K = torch.cat(K, dim=1)
        Q = Q.view(batch_size, -1, 1, hid_dim).transpose(1, 2)
        K = K.view(batch_size, -1, 1, hid_dim).transpose(1, 2)
        V = K
        Q, K, V = [l(x).view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
                   for l, x in zip(self.linear_layers, (Q, K, V))]
        # print(Q[0],K.transpose(-2, -1)[0])
        message_interaction = torch.matmul(Q, K.transpose(-2, -1))/self.d_k
        # print(message_interaction[0])
        att_score = torch.nn.functional.softmax(message_interaction, dim=-1)
        motif_messages = torch.matmul(att_score, V).transpose(
            1, 2).contiguous().view(batch_size, -1, hid_dim)
        motif_messages = self.W_o(motif_messages)
        return motif_messages.squeeze(1)


class Framework(nn.Module):
    def __init__(self,
                 out_dim: int,
                 args_dict,
                 ):
        super(Framework, self).__init__()
        self.args = args_dict

        self.task_num = out_dim

        self.atom_encoder = nn.Sequential(
            LinearBn(args_dict['atom_in_dim'], args_dict['hid_dim']),
            nn.ReLU(inplace=True),
            nn.Dropout(p=args_dict['drop_rate']),
            LinearBn(args_dict['hid_dim'], args_dict['hid_dim']),
            nn.ReLU(inplace=True)
        )
        self.motif_encoder = nn.Sequential(
            LinearBn(args_dict['ss_node_in_dim'], args_dict['hid_dim']),
            nn.ReLU(inplace=True),
            nn.Dropout(p=args_dict['drop_rate']),
            LinearBn(args_dict['hid_dim'], args_dict['hid_dim']),
            nn.ReLU(inplace=True)
        )
        self.step = args_dict['step']
        self.agg_op = args_dict['agg_op']
        self.mol_FP = args_dict['mol_FP']
        # define the message passing layer
        self.motif_mp_layer = DGLMessagePassingNetwork(hid_dim=args_dict['hid_dim'], edge_func=nn.Linear(
            args_dict['ss_edge_in_dim'], args_dict['hid_dim'] * args_dict['hid_dim']), residual=args_dict['residual'], num_layers=args_dict['motif_mpnn_layers'])
        self.atom_mp_layer = DGLMessagePassingNetwork(hid_dim=args_dict['hid_dim'], edge_func=nn.Linear(
            args_dict['bond_in_dim'], args_dict['hid_dim'] * args_dict['hid_dim']), residual=args_dict['residual'], num_layers=args_dict['atom_mpnn_layers'])

        # define the update function
        self.motif_update = nn.GRUCell(
            args_dict['hid_dim'], args_dict['hid_dim'])
        self.atom_update = nn.GRUCell(
            args_dict['hid_dim'], args_dict['hid_dim'])

        # define the readout layer
        self.atom_readout = Readout(
            args_dict, ntype='atom', use_attention=args_dict['attention'])
        self.motif_readout = Readout(
            args_dict, ntype='func_group', use_attention=args_dict['attention'])
        self.LA = AttnLocalAugmentation(args_dict)
        # define the predictor
        atom_MLP_inDim = args_dict['hid_dim'] * 2
        Motif_MLP_inDim = args_dict['hid_dim'] * 2
        if self.mol_FP == 'atom':
            atom_MLP_inDim = atom_MLP_inDim+args_dict['mol_in_dim']
        elif self.mol_FP == 'ss':
            Motif_MLP_inDim = Motif_MLP_inDim+args_dict['mol_in_dim']
        elif self.mol_FP == 'both':
            atom_MLP_inDim = atom_MLP_inDim+args_dict['mol_in_dim']
            Motif_MLP_inDim = Motif_MLP_inDim+args_dict['mol_in_dim']

        self.prompter = AdaptivePrompter(
            hidden_dim=atom_MLP_inDim, task_num=self.task_num, args_dict=args_dict)

        self.out_af = MLP(atom_MLP_inDim,
                          out_dim,
                          dropout_prob=args_dict['drop_rate'],
                          num_neurons=args_dict['num_neurons'], input_norm=args_dict['input_norm'])
        self.out_ff = MLP(Motif_MLP_inDim,
                          out_dim,
                          dropout_prob=args_dict['drop_rate'],
                          num_neurons=args_dict['num_neurons'], input_norm=args_dict['input_norm'])

        # Save the init_type for later use in _init_weights
        self.init_type = args_dict['init_type']
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if self.init_type == 'normal':
                # Normal initialization
                init.normal_(m.weight, mean=0, std=0.01)
            elif self.init_type == 'he':
                # He initialization
                init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif self.init_type == 'xavier':
                # Xavier initialization
                init.xavier_uniform_(m.weight)

            if m.bias is not None:
                init.constant_(m.bias, 0)

    def forward(self, g):
        with g.local_scope():
            af = g.nodes['atom'].data['feat']
            bf = g.edges[('atom', 'interacts', 'atom')].data['feat']
            fnf = g.nodes['func_group'].data['feat']
            fef = g.edges[('func_group', 'interacts',
                           'func_group')].data['feat']
            mf = g.nodes['molecule'].data['feat']

            ufnf = self.motif_encoder(fnf)
            uaf = self.atom_encoder(af)

            for _ in range(self.step):
                ufnm = self.motif_mp_layer(
                    g[('func_group', 'interacts', 'func_group')], ufnf, fef)
                uam = self.atom_mp_layer(
                    g[('atom', 'interacts', 'atom')], uaf, bf)
                g.nodes['atom'].data['_uam'] = uam
                if self.agg_op == 'sum':
                    g.update_all(dgl.function.copy_u('_uam', 'uam'), dgl.function.sum('uam', 'agg_uam'),
                                 etype=('atom', 'a2f', 'func_group'))
                elif self.agg_op == 'max':
                    g.update_all(dgl.function.copy_u('_uam', 'uam'), dgl.function.max('uam', 'agg_uam'),
                                 etype=('atom', 'a2f', 'func_group'))
                elif self.agg_op == 'mean':
                    g.update_all(dgl.function.copy_u('_uam', 'uam'), dgl.function.mean('uam', 'agg_uam'),
                                 etype=('atom', 'a2f', 'func_group'))

                augment_ufnm = g.nodes['func_group'].data['agg_uam']

                ufnm = self.LA(augment_ufnm, ufnm, ufnf)

                ufnf = self.motif_update(ufnm, ufnf)
                uaf = self.atom_update(uam, uaf)

            motif_readout = self.motif_readout(g, ufnf)
            atom_readout = self.atom_readout(g, uaf)

            atom_representation = atom_readout
            motif_representation = motif_readout

            if self.mol_FP == 'atom':
                atom_representation = torch.cat([atom_readout, mf], dim=1)
            elif self.mol_FP == 'ss':
                motif_representation = torch.cat([motif_readout, mf], dim=1)
            elif self.mol_FP == 'both':
                atom_representation = torch.cat([atom_readout, mf], dim=1)
                motif_representation = torch.cat([motif_readout, mf], dim=1)

            atom_pred, motif_pred = None, None
            motif_pred = self.prompter(motif_representation)
            atom_pred = self.prompter(atom_representation)

            return atom_pred, motif_pred
