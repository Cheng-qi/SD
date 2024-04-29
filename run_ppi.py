#!/usr/bin/env python
# encoding: utf-8

"""
@version: 3.7
@author: Qi Cheng
@contact: chengqi@hrbeu.edu.cn
@site: https://github.com/Cheng-qi
@software: PyCharm
@file: main.py
@time: 2020/3/25 9:29
"""
from copy import deepcopy
import scipy.sparse as sp
import torch
import numpy as np
import torch.nn.functional as F
# from model_integration import IntegrationGNN
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, PPI
import torch_geometric.transforms as T
# from GNNModels import *
from models.SD_GNN import *
from sklearn.metrics import accuracy_score, f1_score
import argparse
from torch_geometric.typing import SparseTensor
import random
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--data_name', default='PPI', help='dateset')
parser.add_argument('--split', default='full', help='split method')
parser.add_argument('--patience', type=int, default=500, help='Patience')
parser.add_argument('--max_epoch', type=int, default=5000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
# parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay (L2 loss on parameters).')
parser.add_argument('--weight_decay1', type=float, default=0.00001, help='weight decay (L2 loss on parameters).')
parser.add_argument('--weight_decay2', type=float, default=0.00001, help='weight decay (L2 loss on parameters).')
parser.add_argument('--log_interval', type=int, default=1)
parser.add_argument('--model_name', default='sdgat', help='model name')
parser.add_argument('--residual_type', default='add', help='redusial')
parser.add_argument('--residual_fusion', default=None, help='redusial')
parser.add_argument('--hidden_size', type=int, default=512, help='hidden dimensions.')
parser.add_argument('--num_layer', type=int, default=2, help='Number of layers.')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
parser.add_argument('--gat_heads', type=int, default=4, help='GAT Heads')
parser.add_argument('--gat_dropout', type=float, default=0.5, help='Gat Dropout rate (1 - keep probability).')
parser.add_argument('--lo_ss_train', type=float, default=0, help='alpha smooth loss for train')
parser.add_argument('--lo_ss_val', type=float, default=0, help='alpha smooth loss for valid')
parser.add_argument('--lo_ss_test', type=float, default=0, help='alpha smooth loss for test')
parser.add_argument('--gin_eps_trainable', action="store_true", default=False)
parser.add_argument('--seed', type=int, default=5, help='Random seed.')
parser.add_argument('--log_dir', type=str, default='./logs3', help='log dir')

args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.makedirs(args.log_dir, exist_ok=True)
log_path = f"{args.log_dir}/{args.data_name}-{args.model_name}.csv"
if not os.path.isfile(log_path):
    with open(log_path, "w") as f:
        f.write(",".join(map(str, vars(args).keys()))+",acc\n")

print(args)
# data_name = args.data_name
train_dataset = PPI(root='./data', transform=T.NormalizeFeatures())
val_dataset = PPI(root='./data', split='val', transform=T.NormalizeFeatures())
test_dataset = PPI(root='./data', split='test', transform=T.NormalizeFeatures())
# for dataset in [train_dataset, val_dataset, test_dataset]:
#     for i,data in enumerate(dataset):
#         dataset[i] = data.to('cuda:0')

if args.model_name == "sdgat":
    Model = SDGAT
elif args.model_name == "sdgcn":
    Model = SDGCN
elif args.model_name == "sdgin":
    Model = SDGIN
elif args.model_name == "gcn":
    Model = GCN
model = Model(
    train_dataset.num_node_features,
    args.hidden_size,
    args.num_layer,
    train_dataset.num_classes,
    dropout=args.dropout,
    residual_type = args.residual_type,
    residual_fusion = args.residual_fusion,
    lo_ss_train = args.lo_ss_train,
    lo_ss_val = args.lo_ss_val,
    lo_ss_test = args.lo_ss_test,
    gat_dropout=args.gat_dropout,
    gat_heads=args.gat_heads,
    gin_eps_trainable=args.gin_eps_trainable
    ).to(device)
# data = data.to(device)
# if args.split == 'public':
optimizer = torch.optim.Adam([
                        {'params':model.params_gnn.parameters(),'weight_decay':args.weight_decay1},
                        {'params':model.params_other.parameters(),'weight_decay':args.weight_decay2},
                        ],lr=args.lr) 
# else:
#     optimizer = torch.optim.Adam(model.parameters(), 
#                                 lr=args.lr, 
#                                 weight_decay=args.weight_decay)
# optimizer = torch.optim.Adam(model.parameters(), 
#                              lr=args.lr, 
#                              weight_decay=args.weight_decay)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

model.train()
max_val_acc = 0

cur_step = 0
def train(dataset, model, optimizer):
    outs = []
    labels = []
    losss = []
    model.train()
    for data in dataset:
        data = data.to('cuda:0')
        x = data.x[:(data.edge_index.max()+1)]
        y = (data.y[:(data.edge_index.max()+1)]).argmax(-1)
        optimizer.zero_grad()
        outs_batch, _ = model(x, data.edge_index)
        loss = model.loss(outs_batch, y.clone(), train_mask = torch.full(y.shape, True).to(y.device), val_mask = torch.full(y.shape, False).to(y.device), test_mask = torch.full(y.shape, False).to(y.device))
        loss.backward()
        optimizer.step()
        outs.append(outs_batch[0].cpu().data.numpy().argmax(-1))
        labels.append(y.data.cpu().numpy())
        losss.append(loss.data.cpu().numpy())
    outs = np.concatenate(outs, 0)
    labels = np.concatenate(labels, 0)
    return sum(losss)/len(losss), f1_score(outs, labels, average='weighted')

def val(dataset, model):
    model.eval()
    outs = []
    labels = []
    with torch.no_grad():
        for data in dataset:
            data = data.to('cuda:0')
            x = data.x[:(data.edge_index.max()+1)]
            y = (data.y[:(data.edge_index.max()+1)]).argmax(-1)
            outs_batch, _ = model(x, data.edge_index)
            outs.append(outs_batch[0].cpu().data.numpy().argmax(-1))
            labels.append(y.data.cpu().numpy())
    outs = np.concatenate(outs, 0)
    labels = np.concatenate(labels, 0)
    return f1_score(outs, labels, average='weighted')

for epoch in range(args.max_epoch):
    loss, train_acc = train(train_dataset, model, optimizer)
    if (epoch+1) % 1==0:
        val_acc = val(val_dataset, model)
        test_acc = val(test_dataset, model)
        model.eval()
        if (epoch+1) % args.log_interval==0:
            print(f"epoch{epoch+1}: train_loss={round(loss, 5)}, train_f1={round(train_acc * 100 ,2)}, val_f1={round(val_acc * 100,2)}, test_f1={round(test_acc * 100,2)}")
        if(val_acc > max_val_acc):
            max_val_acc = val_acc
            best_test_acc = test_acc
            cur_step = 0
            print(f"epoch{epoch+1}: update best result: val_f1={round(val_acc * 100, 2)}, test_f1={round(test_acc * 100,2)}")
        else:
            cur_step +=1
            if(cur_step>args.patience):
                print("early stop...")
                break

print('Test Accuracy: {:.8f}'.format(best_test_acc))
with open(log_path, "a") as f:
    f.write(",".join(map(str, vars(args).values()))+",%.2f\n"%(best_test_acc*100))



