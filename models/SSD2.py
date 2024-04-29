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
import math
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
        # x=F.normalize(x)
        x_mean = spmm(adj, x, "mean")
        x_std = spmm(adj, (x-x_mean)**2, "mean")
        # score = self.weight_score_func(torch.cat([x_mean, x_std, x], -1))
        score = self.weight_score_func(torch.cat([x_mean*x, x_std, x], -1))
        # score = self.weight_score_func(torch.cat([x_mean * x, x_std], -1))
        # score = self.weight_score_func(x_mean * x)
        # score = self.weight_score_func(x_mean * x+x_std)
        # score = torch.sigmoid((x_mean * x).mean())
        # score = torch.sigmoid((x_mean * x).sum(-1, True))
        return score
# pygnn.GCN
class SSD(nn.Module):
    def __init__(self) -> None:
        super().__init__()
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
            # x = self.layer_norm_layers[i](x)
            x = F.normalize(x)
            if self.use_ss_loss:
                hs.append(x)
            if self.residual_type == "de-smoothing":
                weight_score = self.weightScore(x, adj)
                if self.residual_fusion == "cat":
                    x = torch.cat([(1-weight_score) * x, weight_score * F.normalize(x0)], -1)
                else:
                    x = (1-weight_score) * x + weight_score * F.normalize(x0)
                    # x = x + weight_score * F.normalize(x0)
            elif self.residual_type == "add":
                x += x0
            # x = F.normalize(x)
            # if self.use_ss_loss:
            #     hs.append(x)
            x = self.dropout(x)
        return [x, hs, adj], log
    def ss_loss(self, h, y, adj,  mask, alphas):
        with torch.no_grad():
            adj = adj[mask, mask]
            adj = adj.fill_value(1.0)
            adj_2 = spmm(adj, adj.t()).fill_diag(0).to_dense()

            y = y[mask]
            # h_mat = h[:]
            y_mat = F.one_hot(y).to(dtype=torch.float32)
            y_y_hat_intra = adj_2 * ( y_mat @ y_mat.t())
            y_y_hat_inter = adj_2 * (1- y_mat @ y_mat.t())

            y_y_hat_intra = torch.triu(y_y_hat_intra, diagonal=0)
            y_y_hat_inter = torch.triu(y_y_hat_inter, diagonal=0)
            # y_y_hat = torch.triu(adj_2 * (1- y_mat @ y_mat.t()), diagonal=0)
            # y_y_hat = torch.triu(adj_2 * (y_mat @ y_mat.t()), diagonal=0)
            # y_y_hat_inter = torch.triu(adj_2 * (1- y_mat @ y_mat.t()), diagonal=0)
            # y_y_hat_intra = torch.triu(adj_2 * (1- y_mat @ y_mat.t()), diagonal=0)


            # rows, cols = torch.where(y_y_hat !=0)
            rows_inter, cols_inter = torch.where(y_y_hat_inter !=0)
            rows_intra, cols_intra = torch.where(y_y_hat_intra >0)
        ss_1 = ((h[rows_inter] * h[cols_inter] * alphas[rows_inter].view(-1,1)*alphas[cols_inter].view(-1,1)).sum(-1) * y_y_hat_inter[rows_inter, cols_inter]).sum()/y_y_hat_inter.sum() \
        -((h[rows_intra] * h[cols_intra] * alphas[rows_intra].view(-1,1)*alphas[cols_intra].view(-1,1)).sum(-1) * y_y_hat_intra[rows_intra, cols_intra]).sum()/y_y_hat_intra.sum()
        # ss_all = 0
        # if last_h !=None:
        #     ss_all = -(((h[rows_inter] - h[cols_inter])**2 * alphas[rows_inter].view(-1,1)*alphas[cols_inter].view(-1,1)).sum(-1) * y_y_hat_inter[rows_inter, cols_inter]).sum()/y_y_hat_inter.sum() \
        # +(((h[rows_intra] - h[cols_intra])**2 * alphas[rows_intra].view(-1,1)*alphas[cols_intra].view(-1,1)).sum(-1) * y_y_hat_intra[rows_intra, cols_intra]).sum()/y_y_hat_intra.sum()
        # ss_all = 
        return  ss_1
        
    def loss(self, outs, labels, train_mask = None, val_mask = None, test_mask = None):
        x, hs, adj = outs
        l =  F.cross_entropy(x[train_mask], labels[train_mask]) # 交叉熵
        if self.use_ss_loss:
            # last_h = None
            labeled_mask = torch.ones(train_mask.size(), dtype=torch.bool, device= train_mask.device, requires_grad = False)
            alphas = torch.ones(train_mask.size(),  dtype=torch.float, device= train_mask.device) * self.lo_ss_train
            alphas[train_mask] = self.lo_ss_train
            alphas[val_mask] = self.lo_ss_val
            labels[val_mask] = outs[0][val_mask].detach().argmax(-1)
            labels[test_mask] = outs[0][test_mask].detach().argmax(-1)
            alphas[test_mask] = self.lo_ss_test * F.softmax(outs[0][test_mask].detach(), -1).max(-1)[0]
            for i, h in enumerate(hs[:]):
                # alphas = torch.ones(train_mask.size(),  dtype=torch.float, device= train_mask.device)
                # labeled_mask = torch.zeros(train_mask.size(), dtype=torch.bool, device= train_mask.device, requires_grad = False)
                # labeled_mask.bitwise_xor(train_mask.bitwise_xor(train_mask))
                # if self.lo_ss_train != None: 
                #     labeled_mask = labeled_mask.bitwise_or(train_mask)
                #     alphas[train_mask] = self.lo_ss_train
                # if self.lo_ss_val != None:
                #     labeled_mask = labeled_mask.bitwise_or(val_mask)
                #     # labels[val_mask] = outs[0][val_mask].detach().argmax(-1)
                #     alphas[val_mask] = self.lo_ss_val
                # if self.lo_ss_test != None:
                #     labeled_mask = labeled_mask.bitwise_or(test_mask)
                #     labels[test_mask] = outs[0][test_mask].detach().argmax(-1)
                #     alphas[test_mask] = self.lo_ss_test * F.softmax(outs[0][test_mask].detach(), -1).max(-1)[0]
                    
                l += self.ss_loss(h, labels, adj, labeled_mask, alphas) * math.exp(-(i+1)/len(hs))
                # last_h = h
                if i>0:
                    l += ((h - hs[i-1]).norm(2,1)* alphas).mean() * math.exp(-(i+1)/len(hs)) 
                
                # if self.lo_ss_val != None: l += self.lo_ss_val * self.ss_loss(h, labels, adj, val_mask) 
                # if self.lo_ss_test != None and use_test: l += self.lo_ss_test * self.ss_loss(h, outs[0].argmax(-1), adj, test_mask) 
        return l
class SDGAT(SSD):
    def __init__(
            self,
            in_channels,
            hidden_channels,
            num_layers,
            out_channels,
            dropout = 0.5,
            residual_type = "de-smoothing",
            residual_fusion = "cat",
            lo_ss_train = 0.2,
            lo_ss_val = 0.5,
            lo_ss_test = None,
            gat_heads=4,
            gat_dropout=0.0) -> None:
        super().__init__()
        self.residual_fusion = residual_fusion
        

        self.lo_ss_train = lo_ss_train
        self.lo_ss_val = lo_ss_val
        self.lo_ss_test = lo_ss_test
        self.use_ss_loss = lo_ss_train != 0 or lo_ss_val != 0 or lo_ss_test != 0
        self.residual_type = residual_type
        self.n_layers = num_layers
        in_hidden_size = hidden_channels * gat_heads
        if self.residual_fusion == "cat" and residual_type == "de-smoothing":
            in_hidden_size = in_hidden_size*2
        
        if num_layers == 1:
            self.gnn_layers = nn.ModuleList([pygnn.GATConv(in_channels, out_channels, dropout=gat_dropout, heads=gat_heads)])
        else:
            self.gnn_layers = nn.ModuleList([pygnn.GATConv(in_channels, hidden_channels, dropout=gat_dropout, heads=gat_heads)])
            # if residual_type != None: self.layer_norm_layers = nn.ModuleList()
            for i in range(num_layers-1):
                # if residual_type != None:
                #     self.layer_norm_layers.append(nn.LayerNorm(hidden_size))
                self.gnn_layers.append(pygnn.GATConv(in_hidden_size, hidden_channels if i!=num_layers-2 else out_channels, dropout=gat_dropout, heads=gat_heads))
                if residual_type == "de-smoothing":
                    self.weightScore = WeightScoreLayer(hidden_channels * gat_heads)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        
        if residual_type != None:
            self.residual_fc = nn.Sequential(
                nn.Linear(in_channels, hidden_channels * gat_heads),
                nn.ReLU()
            )
        
class SDGCN(SSD):
    def __init__(
            self,
            in_channels,
            hidden_channels,
            num_layers,
            out_channels,
            dropout = 0.5,
            residual_type = "de-smoothing",
            residual_fusion = "cat",
            lo_ss_train = 0.2,
            lo_ss_val = 0.5,
            lo_ss_test = None,
            **kward) -> None:
        super().__init__()
        self.residual_fusion = residual_fusion
        in_hidden_size = hidden_channels

        self.lo_ss_train = lo_ss_train
        self.lo_ss_val = lo_ss_val
        self.lo_ss_test = lo_ss_test
        self.use_ss_loss = lo_ss_train != 0 or lo_ss_val != 0 or lo_ss_test != 0
        self.residual_type = residual_type
        self.n_layers = num_layers
        if self.residual_fusion == "cat" and residual_type == "de-smoothing":
            in_hidden_size += hidden_channels
        if num_layers == 1:
            self.gnn_layers = nn.ModuleList([pygnn.GCNConv(in_channels, out_channels)])
        else:
            self.gnn_layers = nn.ModuleList([pygnn.GCNConv(in_channels, hidden_channels)])
            # if residual_type != None: self.layer_norm_layers = nn.ModuleList()
            for i in range(num_layers-1):
                # if residual_type != None:
                #     self.layer_norm_layers.append(nn.LayerNorm(hidden_size))
                self.gnn_layers.append(pygnn.GCNConv(in_hidden_size, hidden_channels if i!=num_layers-2 else out_channels))
                if residual_type == "de-smoothing":
                    self.weightScore = WeightScoreLayer(hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        
        if residual_type != None:
            self.residual_fc = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU()
            )


class SDGIN(SSD):
    def __init__(
            self,
            in_channels,
            hidden_channels,
            num_layers,
            out_channels,
            dropout = 0.5,
            residual_type = "de-smoothing",
            residual_fusion = "cat",
            lo_ss_train = 0.2,
            lo_ss_val = 0.5,
            lo_ss_test = None,
            gin_eps_trainable = False, 
            **kward) -> None:
        super().__init__()
        self.residual_fusion = residual_fusion
        in_hidden_size = hidden_channels

        self.lo_ss_train = lo_ss_train
        self.lo_ss_val = lo_ss_val
        self.lo_ss_test = lo_ss_test
        self.use_ss_loss = lo_ss_train != 0 or lo_ss_val != 0 or lo_ss_test != 0
        self.residual_type = residual_type
        self.n_layers = num_layers
        if self.residual_fusion == "cat" and residual_type == "de-smoothing":
            in_hidden_size += hidden_channels
        if num_layers == 1:
            self.gnn_layers = nn.ModuleList([pygnn.GINConv(nn.Linear(in_channels, out_channels), train_eps=gin_eps_trainable)])
        else:
            self.gnn_layers = nn.ModuleList([pygnn.GINConv(nn.Linear(in_channels, hidden_channels), train_eps=gin_eps_trainable)])
            # if residual_type != None: self.layer_norm_layers = nn.ModuleList()
            for i in range(num_layers-1):
                # if residual_type != None:
                #     self.layer_norm_layers.append(nn.LayerNorm(hidden_size))
                self.gnn_layers.append(pygnn.GINConv(nn.Linear(in_hidden_size, hidden_channels if i!=num_layers-2 else out_channels), train_eps=gin_eps_trainable))
                if residual_type == "de-smoothing":
                    self.weightScore = WeightScoreLayer(hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        
        if residual_type != None:
            self.residual_fc = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU()
            )
    # def mlp(self, in_channels, out_channels):
    #     in_channels



# pygnn.GCN
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
