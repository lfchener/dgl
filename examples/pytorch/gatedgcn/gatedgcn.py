import torch
from torch import nn
import dgl.function as fn
import sys
import dgl
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

class GatedGCNLayer(nn.Module):
    """
        Param: []
    """
    def __init__(self, input_dim, output_dim, dropout, batch_norm, residual=False, edge_fea=False):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        self.edge_fea = edge_fea
        
        if input_dim != output_dim:
            self.residual = False
        
        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.D = nn.Linear(input_dim, output_dim, bias=True)
        self.E = nn.Linear(input_dim, output_dim, bias=True)
        self.bn_node_h = nn.BatchNorm1d(output_dim)

        if self.edge_fea:
            self.C = nn.Linear(input_dim, output_dim, bias=True)
            self.bn_node_e = nn.BatchNorm1d(output_dim)
    
    def forward(self, g):
        h = g.ndata['h']
        h_in = h # for residual connection
        
        g.ndata['Ah'] = self.A(h) 
        g.ndata['Bh'] = self.B(h) 
        g.ndata['Dh'] = self.D(h)
        g.ndata['Eh'] = self.E(h) 

        if self.edge_fea:
            e = g.edata['e']
            e_in = e # for residual connection
            g.edata['Ce'] = self.C(e) 
        
        g.apply_edges(fn.u_add_v('Dh', 'Eh', 'DEh'))
        if self.edge_fea:
            g.edata['e'] = g.edata['DEh'] + g.edata['Ce']
        else:
            g.edata['e'] = g.edata['DEh']
        g.edata['sigma'] = torch.sigmoid(g.edata['e'])
        g.update_all(fn.u_mul_e('Bh', 'sigma', 'm'), fn.sum('m', 'sum_sigma_h'))
        g.update_all(fn.copy_e('sigma', 'm'), fn.sum('m', 'sum_sigma'))
        g.ndata['h'] = g.ndata['Ah'] + g.ndata['sum_sigma_h'] / (g.ndata['sum_sigma'] + 1e-6)
        #g.update_all(self.message_func,self.reduce_func) 
        h = g.ndata['h'] # result of graph convolution
        e = g.edata['e'] # result of graph convolution
        
        if self.batch_norm:
            h = self.bn_node_h(h) # batch normalization  
        
        h = F.relu(h) # non-linear activation
        
        if self.residual:
            h = h_in + h # residual connection
        
        h = F.dropout(h, self.dropout, training=self.training)
        
        if self.edge_fea:
            if self.batch_norm:  
                e = self.bn_node_e(e) # batch normalization  
            e = F.relu(e) # non-linear activation
            if self.residual:
                e = e_in + e # residual connection
            e = F.dropout(e, self.dropout, training=self.training)
        
        return h, e

class GatedGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, L, dropout, batch_norm, residual, edge_fea = False):
        super().__init__()
        self.embedding_h = nn.Linear(input_dim, hidden_dim)
        self.embedding_e = nn.Linear(1, hidden_dim)

        self.GatedGCN_layers = nn.ModuleList()
        if L > 1:
            self.layers.append(GatedGCNLayer(input_dim, hidden_dim, dropout, batch_norm, residual))
            for i in range(1, L - 1):
                self.GatedGCN_layers.append(GatedGCNLayer(hidden_dim, hidden_dim, dropout, batch_norm, residual))
            self.GatedGCN_layers.append(GatedGCNLayer(hidden_dim, output_dim, dropout, batch_norm, residual))
        else:
            self.GatedGCN_layers.append(GatedGCNLayer(input_dim, output_dim, dropout, batch_norm, residual))
            
        # self.GatedGCN_layers = nn.ModuleList([
        #     GatedGCNLayer(hidden_dim, hidden_dim, dropout, batch_norm, residual) for _ in range(L)
        # ])
        self.MLP_layer = MLPReadout(hidden_dim, output_dim)
        self.edge_fea = edge_fea
    def forward(self, g, h, e = None):
        # input embedding
        #h = self.embedding_h(h)
        if self.edge_fea and e:
            e = self.embedding_e(e)
        # graph convnet layers
        for GGCN_layer in self.GatedGCN_layers:
            g.ndata['h'] = h
            if self.edge_fea:
                g.edata['e'] = e
            h, e = GGCN_layer(g)
        # MLP classifier
        y = self.MLP_layer(h)
        return y