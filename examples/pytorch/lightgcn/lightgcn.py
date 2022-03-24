"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import torch
import torch.nn as nn
import dgl.function as fn


class LightGCN(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_classes):
        super(LightGCN, self).__init__()
        self.num_layers = num_layers
        self.fc = nn.Linear(in_dim, num_classes , bias=True)
        

    def forward(self, graph, inputs):
        outputs = []
        graph.ndata['h'] = inputs
        for l in range(self.num_layers):
            graph.update_all(fn.copy_src('h', 'm'), fn.sum(msg='m', out='h'))
            outputs.append(graph.ndata['h'])
        # output projection
        light_out = torch.mean(torch.stack(outputs, dim=1), dim=1)
        logits = self.fc(light_out)
        return logits
