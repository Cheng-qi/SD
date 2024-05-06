#!/usr/bin/env python
# encoding: utf-8

import torch
from torch import nn
import random
import torch.nn.functional as F
import torch_geometric.nn as pygnn
import math

from torch import Tensor
from torch_geometric.typing import (
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
        score = self.weight_score_func(torch.cat([x_mean, x_std, x], -1))
        return score
# pygnn.GCN
class SSD(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.epoch=0
    def forward(self, x, edge_index):
        log={}
        if not isinstance(edge_index, SparseTensor):
            adj = SparseTensor.from_edge_index(edge_index)
        else:
            adj = edge_index
        if self.residual_type != None:
            x0 = self.residual_fc(x)
            x0 = self.dropout(x0)
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
                x = x + x0
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
            rows_inter, cols_inter = torch.where(y_y_hat_inter >0)
            rows_intra, cols_intra = torch.where(y_y_hat_intra >0)
            if rows_inter.shape[0]>100000:
                random_inter = torch.LongTensor(random.sample(range(rows_inter.shape[0]), int(100000)))
                rows_inter = rows_inter[random_inter]
                cols_inter = cols_inter[random_inter]
            if rows_intra.shape[0]>100000:
                random_intar = torch.LongTensor(random.sample(range(rows_intra.shape[0]), int(100000)))
                rows_intra = rows_intra[random_intar]
                cols_intra = cols_intra[random_intar]
        ss_1 = (( h[rows_inter] * h[cols_inter] * torch.sqrt(alphas[rows_inter]).view(-1,1) * torch.sqrt(alphas[cols_inter]).view(-1,1)).sum(-1) * y_y_hat_inter[rows_inter, cols_inter]).sum()/(adj_2.sum()/2) \
        +   ((1/(1+((h[rows_intra] * h[cols_intra])* torch.sqrt(alphas[rows_intra]).view(-1,1)*torch.sqrt(alphas[cols_intra]).view(-1,1)).sum(-1))) * y_y_hat_intra[rows_intra, cols_intra]).sum()/(adj_2.sum()/2)
        return  ss_1
        
    def loss(self, outs, labels, train_mask = None, val_mask = None, test_mask = None):
        self.epoch +=1
        x, hs, adj = outs
        # if self.epoch%50<25 or self.epoch<30:
        l =  F.cross_entropy(x[train_mask], labels[train_mask]) # 交叉熵
        # else:
            # l=0
        # if self.use_ss_loss and self.epoch>100:
        if self.use_ss_loss:
            # last_h = None
            labeled_mask = torch.ones(train_mask.size(), dtype=torch.bool, device= train_mask.device, requires_grad = False)
            alphas = torch.ones(train_mask.size(),  dtype=torch.float, device= train_mask.device) * self.lo_ss_train
            alphas[train_mask] = self.lo_ss_train
            alphas[torch.logical_not(train_mask)] = self.lo_ss_val

            # alphas[val_mask] = self.lo_ss_val
            alphas[test_mask] = self.lo_ss_test 
            if not (torch.logical_not(train_mask)==False).all():
                labels[torch.logical_not(train_mask)] = outs[0][torch.logical_not(train_mask)].detach().argmax(-1)
            for i, h in enumerate(hs[2:-1]):
                l += (self.ss_loss(h, labels, adj, labeled_mask, alphas) + (hs[i]- hs[i-1]).norm()/x.shape[0]) * math.exp((i)/len(hs))
        return l
    def getSMV(cls, outs, labels, **kward):
        x, hs, adj = outs
        node_embeding = hs[-1]
        # 全平滑
        smv_matrix = 1 - node_embeding @ node_embeding.t()
        smv = smv_matrix.mean()
        
        y_mat = F.one_hot(labels).to(dtype=torch.float32)
        y_y_hat_intra = y_mat @ y_mat.t()
        y_y_hat_inter = 1- y_y_hat_intra

        # y_y_hat_intra = torch.triu(y_y_hat_intra, diagonal=0)
        # y_y_hat_inter = torch.triu(y_y_hat_inter, diagonal=0)

        # 类内平滑
        smv_intra = (smv_matrix * y_y_hat_intra).sum()/y_y_hat_intra.sum()
        smv_inter = (smv_matrix * y_y_hat_inter).sum()/y_y_hat_inter.sum()
        # 类间平滑
        return smv.item(), smv_intra.item(), smv_inter.item()
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
            gat_dropout=0.0,
            **kward) -> None:
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
                    self.params_other =nn.ModuleList([self.weightScore])
                else:
                    self.params_other =nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

        self.params_gnn =self.gnn_layers
        if residual_type != None:
            self.residual_fc = nn.Sequential(
                nn.Linear(in_channels, hidden_channels * gat_heads),
                nn.ReLU()
            )
            self.params_other.append(self.residual_fc)
        
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
            for i in range(num_layers-1):
                self.gnn_layers.append(pygnn.GCNConv(in_hidden_size, hidden_channels if i!=num_layers-2 else out_channels))
                if residual_type == "de-smoothing":
                    self.weightScore = WeightScoreLayer(hidden_channels)
                    self.params_other =nn.ModuleList([self.weightScore])
                else:
                    self.params_other =nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.params_gnn =self.gnn_layers
        
        if residual_type != None:
            self.residual_fc = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU()
            )
            self.params_other.append(self.residual_fc)
        


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
            for i in range(num_layers-1):
                self.gnn_layers.append(pygnn.GINConv(nn.Linear(in_hidden_size, hidden_channels if i!=num_layers-2 else out_channels), train_eps=gin_eps_trainable))
                if residual_type == "de-smoothing":
                    self.weightScore = WeightScoreLayer(hidden_channels)
                    self.params_other =nn.ModuleList([self.weightScore])
                else:
                    self.params_other =nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        
        self.params_gnn =self.gnn_layers
        if residual_type != None:
            self.residual_fc = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU()
            )
            self.params_other.append(self.residual_fc)




