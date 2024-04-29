
from typing import Any, Dict, List, Optional, Union
import torch
from torch import nn
import random
import torch.nn.functional as F
import torch_geometric.nn as pygnn
from .GCNII.model import GCNII as GCNIIUnit
import networkx as nx
import numpy as np
import scipy.sparse as sp
from torch_geometric.typing import SparseTensor
# pygnn.GCN
def sys_normalized_adjacency(adj):
   adj = sp.coo_matrix(adj)
   adj = adj + sp.eye(adj.shape[0])
   row_sum = np.array(adj.sum(1))
   row_sum=(row_sum==0)*1+row_sum
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
def parse_adj(edge_index):
    # adj = nx.adjacency_matrix(SparseTensor.from_edge_index(edge_index).to_torch_sparse_coo_tensor().cpu())
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = sys_normalized_adjacency(SparseTensor.from_edge_index(edge_index).to_scipy())
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj

class GCN(torch.nn.Module):
    def __init__(self,
            in_channels,
            hidden_channels,
            n_layer,
            n_classes,
            dropout = 0.5,
            **kward):
        super().__init__()
        self.gnn = pygnn.GCN(in_channels, hidden_channels, n_layer, n_classes, dropout=dropout, act='relu')
        
    def forward(self, x, edge_index):
        return self.gnn(x, edge_index), None
    def loss(self, outs, labels, train_mask = None, **kward):
        return F.cross_entropy(outs[train_mask], labels[train_mask])
    
class GRAPHSAGE(torch.nn.Module):
    def __init__(self,
            in_channels,
            hidden_channels,
            n_layer,
            n_classes,
            dropout = 0.5,
            **kward):
        super().__init__()
        self.gnn = pygnn.GraphSAGE(in_channels, hidden_channels, n_layer, n_classes, dropout=dropout, act='relu')
        
    def forward(self, x, edge_index):
        return self.gnn(x, edge_index), None
    def loss(self, outs, labels, train_mask = None, **kward):
        return F.cross_entropy(outs[train_mask], labels[train_mask])
    
class GAT(torch.nn.Module):
    def __init__(self,
            in_channels,
            hidden_channels,
            n_layer,
            n_classes,
            dropout = 0.5,
            gat_heads = 8,
            **kward):
        super().__init__()
        self.gnn = pygnn.GAT(in_channels, hidden_channels, 
                             n_layer,n_classes, dropout, heads=gat_heads, act='elu')
    def forward(self, x, edge_index):
        return self.gnn(x, edge_index), None
    def loss(self, outs, labels, train_mask = None, **kward):
        return F.cross_entropy(outs[train_mask], labels[train_mask])
    
class GIN(torch.nn.Module):
    def __init__(self,
            in_channels,
            hidden_channels,
            n_layer,
            n_classes,
            dropout = 0.5,
            gin_eps_trainable=True,
            **kward):
        super().__init__()
        self.gnn = pygnn.GIN(in_channels, hidden_channels, 
                             n_layer, n_classes, dropout, train_eps=gin_eps_trainable)
    def forward(self, x, edge_index):
        return self.gnn(x, edge_index), None
    def loss(self, outs, labels, train_mask = None, **kward):
        return F.cross_entropy(outs[train_mask], labels[train_mask])
class GCNII(torch.nn.Module):
    def __init__(self,
            in_channels,
            hidden_channels,
            n_layer,
            n_classes,
            dropout = 0.5,
            **kward):
        super().__init__()
        self.gnn = GCNIIUnit(in_channels,
                             n_layer, 
                             hidden_channels, 
                             n_classes,
                             dropout,
                             0.5, 
                             0.2, ## citeseer pubmed 0.1, cora:0.2
                             True)
    def forward(self, x, edge_index):
        return self.gnn(x, edge_index), None
    def loss(self, outs, labels, train_mask = None, **kward):
        return F.cross_entropy(outs[train_mask], labels[train_mask])
