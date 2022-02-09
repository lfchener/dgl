from torch import nn
import dgl
from ....python.dgl.nn.pytorch.conv.gatedgcn import GatedGCNLayer
from dgl.nn import GatedGCNLayer
import torch.nn.functional as F

"""
    MLP Layer used after graph vector representation
"""

class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2): #L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [ nn.Linear( input_dim//2**l , input_dim//2**(l+1) , bias=True ) for l in range(L) ]
        list_FC_layers.append(nn.Linear( input_dim//2**L , output_dim , bias=True ))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y

class GatedGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, L, dropout, batch_norm, residual, edge_fea = False):
        super().__init__()
        self.embedding_h = nn.Linear(input_dim, hidden_dim)
        self.embedding_e = nn.Linear(1, hidden_dim)
        self.GatedGCN_layers = nn.ModuleList([
            GatedGCNLayer(hidden_dim, hidden_dim, dropout, batch_norm, residual) for _ in range(L)
        ])
        self.MLP_layer = MLPReadout(hidden_dim, output_dim)
        self.edge_fea = edge_fea
    def forward(self, g, h, e = None):
        # input embedding
        h = self.embedding_h(h)
        if self.edge_fea and e:
            e = self.embedding_e(e)
        # graph convnet layers
        for GGCN_layer in self.GatedGCN_layers:
            g.ndata['h'] = h
            if self.edge_fea:
                g.edata['e'] = e
            h, e = GGCN_layer(g)
        # MLP classifier
        #g.ndata['h'] = h
        #y = dgl.mean_nodes(g,'h')
        y = self.MLP_layer(h)
        return y