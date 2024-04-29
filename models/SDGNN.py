#!/usr/bin/env python
# encoding: utf-8

from typing import Any, Dict, List, Optional, Union
import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn
# from torch_geometric.nn import GATConv
# from torch_geometric.nn import GINConv
from functools import partial

from torch_geometric.nn.aggr import Aggregation 

from torch import Tensor
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
)
from torch_geometric.utils import spmm


class WeightScoreLayer(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.weight_score_func = nn.Sequential(
                        nn.Linear( embedding_size * 3, 1, False),
                        nn.Sigmoid()
                    )

    def forward(self, x, adj) -> Tensor:
        x_mean = spmm(adj, x, "mean")
        x_std = spmm(adj, (x-x_mean).abs(), "mean")
        # score = self.weight_score_func(torch.cat([x_mean, x_std, x], -1))
        score = self.weight_score_func(torch.cat([x_mean*x, x_std, x], -1))
        # score = self.weight_score_func(torch.cat([x_mean * x, x_std], -1))
        # score = self.weight_score_func(x_mean * x)
        # score = self.weight_score_func(x_mean * x+x_std)
        # score = torch.sigmoid((x_mean * x).mean())
        # score = torch.sigmoid((x_mean * x).sum(-1, True))
        return score


class SDGCN(nn.Module):
    def __init__(
            self,
            input_size,
            n_classes,
            hidden_size = 32,
            n_layers = 64,
            dropout = 0.5,
            residual_type = "de-smoothing",
            lo_ss_train = 0.2,
            lo_ss_val = 0.5,
            lo_ss_test = None) -> None:
        super().__init__()
        self.lo_ss_train = lo_ss_train
        self.lo_ss_val = lo_ss_val
        self.lo_ss_test = lo_ss_test
        self.use_ss_loss = lo_ss_train != None or lo_ss_val != None or lo_ss_test != None
        self.residual_type = residual_type
        self.n_layers = n_layers
        if n_layers == 1:
            self.gnn_layers = nn.ModuleList([pygnn.GCNConv(input_size, n_classes)])
        else:
            self.gnn_layers = nn.ModuleList([pygnn.GCNConv(input_size, hidden_size)])
            if residual_type != None: self.layer_norm_layers = nn.ModuleList()
            for i in range(n_layers-1):
                if residual_type != None:
                    self.layer_norm_layers.append(nn.LayerNorm(hidden_size))
                self.gnn_layers.append(pygnn.GCNConv(hidden_size, hidden_size if i!=n_layers-2 else n_classes))
                if residual_type == "de-smoothing":
                    self.weightScore = WeightScoreLayer(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        
        if residual_type != None:
            self.residual_fc = nn.Sequential(
                nn.Linear(input_size, hidden_size, False),
                nn.ReLU(),
                nn.LayerNorm(hidden_size)
            )

    def forward(self, x, edge_index):
        log={}
        if not isinstance(edge_index, SparseTensor):
            adj = SparseTensor.from_edge_index(edge_index)
        else:
            adj = edge_index
        if self.residual_type != None:
            x0 = self.residual_fc(x)
        hs = []
        for i, conv in enumerate(self.gnn_layers):
            x = conv(x, edge_index)
            if i == len(self.gnn_layers)-1:
                break
            x = self.act(x)
            x = self.layer_norm_layers[i](x)
            if self.use_ss_loss:
                hs.append(x)
            if self.residual_type == "de-smoothing":
                weight_score = self.weightScore(x, adj)
                x = (1-weight_score) * x + weight_score * x0
            elif self.residual_type == "add":
                x += x0

            x = self.dropout(x)
        return [x, hs], log
    def ss_loss(self, h, y, mask):
        with torch.no_grad():
            y = y[mask]
            h_mat = h[mask]
            y_mat = F.one_hot(y).to(dtype=torch.float32)
            y_y_hat = 1-y_mat @ y_mat.T

        return (y_y_hat * (h_mat @ h_mat.T)).mean()

        
    def loss(self, outs, labels, train_mask = None, val_mask = None, test_mask = None, use_test = False):
        x, hs = outs
        l =  F.cross_entropy(x[train_mask], labels[train_mask]) # 交叉熵
        if self.use_ss_loss:
            for h in hs:
                if self.lo_ss_train != None: l += self.lo_ss_train * self.ss_loss(h, labels, train_mask) 
                if self.lo_ss_val != None: l += self.lo_ss_val * self.ss_loss(h, labels, val_mask) 
                if self.lo_ss_test != None and use_test: l += self.lo_ss_test * self.ss_loss(h, outs[0].argmax(-1), test_mask) 
        return l


class GCN(torch.nn.Module):
    def __init__(self,
            input_size,
            n_classes,
            hidden_size,
            **kward):
        super().__init__()
        self.conv1 = pygnn.GCNConv(input_size, hidden_size)
        self.conv2 = pygnn.GCNConv(hidden_size, n_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return [F.log_softmax(x, dim=1), None], None
    def loss(self, outs, labels, train_mask = None, **kward):
        return F.nll_loss(outs[0][train_mask], labels[train_mask])
