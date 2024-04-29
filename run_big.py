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
from sklearn.metrics import accuracy_score
import argparse
from torch_geometric.typing import SparseTensor
import random
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--data_name', default='Photo', help='dateset')
parser.add_argument('--split', default='full', help='split method')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--max_epoch', type=int, default=50000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
# parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay (L2 loss on parameters).')
parser.add_argument('--weight_decay1', type=float, default=0.001, help='weight decay (L2 loss on parameters).')
parser.add_argument('--weight_decay2', type=float, default=0.001, help='weight decay (L2 loss on parameters).')
parser.add_argument('--log_interval', type=int, default=1)
parser.add_argument('--model_name', default='sdgin', help='model name')
parser.add_argument('--residual_type', default='de-smoothing', help='redusial')
parser.add_argument('--residual_fusion', default='add', help='redusial')
parser.add_argument('--hidden_size', type=int, default=256, help='hidden dimensions.')
parser.add_argument('--num_layer', type=int, default=8, help='Number of layers.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--gat_heads', type=int, default=1, help='GAT Heads')
parser.add_argument('--gat_dropout', type=float, default=0.5, help='Gat Dropout rate (1 - keep probability).')
parser.add_argument('--lo_ss_train', type=float, default=0.1, help='alpha smooth loss for train')
parser.add_argument('--lo_ss_val', type=float, default=0.1, help='alpha smooth loss for valid')
parser.add_argument('--lo_ss_test', type=float, default=0.1, help='alpha smooth loss for test')
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
if args.data_name.lower() in ['cora', 'citeseer', 'pubmed']:
    dataset = Planetoid(root='./data', 
                        name=args.data_name,
                        num_train_per_class = 20,
                        split=args.split, 
                        transform=T.NormalizeFeatures())
                        # split=args.split, 
                        # transform=T.NormalizeFeatures())
    data = dataset[0]
elif args.data_name.lower() in ['computers', 'photo']:
    dataset = Amazon(root='./data', 
                        name=args.data_name,
                        transform=T.NormalizeFeatures())
    data = dataset[0]
    data.train_mask = torch.full(data.y.shape, False, dtype=bool)
    data.val_mask = torch.full(data.y.shape, False, dtype=bool)
    data.test_mask = torch.full(data.y.shape, False, dtype=bool)
    train_index = torch.LongTensor(random.sample(range(150), int(20)))
    for label in range(data.y.max()+1):
        label_index = torch.where(data.y==label)[0]
        data.test_mask[label_index[150:]] = True
        data.val_mask[label_index[:150]] = True
        data.val_mask[label_index[train_index]] = False
        data.train_mask[label_index[train_index]] = True

elif args.data_name.lower() in ['cs', 'physics']:
    dataset = Coauthor(root='./data', 
                        name=args.data_name,
                        transform=T.NormalizeFeatures())
    data = dataset[0]
    data.train_mask = torch.full(data.y.shape, False, dtype=bool)
    data.val_mask = torch.full(data.y.shape, False, dtype=bool)
    data.test_mask = torch.full(data.y.shape, False, dtype=bool)
    train_index = torch.LongTensor(random.sample(range(100), int(20)))
    for label in range(data.y.max()+1):
        label_index = torch.where(data.y==label)[0]
        data.test_mask[label_index[100:]] = True
        data.val_mask[label_index[:100]] = True
        data.val_mask[label_index[train_index]] = False
        data.train_mask[label_index[train_index]] = True
        # random.sample(range(235), int(235*0.2)).__len__()


if args.model_name == "sdgat":
    Model = SDGAT
elif args.model_name == "sdgcn":
    Model = SDGCN
elif args.model_name == "sdgin":
    Model = SDGIN
elif args.model_name == "gcn":
    Model = GCN
model = Model(
    dataset.num_node_features,
    args.hidden_size,
    args.num_layer,
    dataset.num_classes,
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
data = data.to(device)
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
for epoch in range(args.max_epoch):
    model.train()
    optimizer.zero_grad()
    # out = model(data)
    outs, _ = model(data.x, data.edge_index)
    loss = model.loss(outs, data.y.clone(), train_mask = data.train_mask, val_mask = data.val_mask, test_mask = data.test_mask)
    loss.backward()
    optimizer.step()
    # scheduler.step()
    if (epoch+1) % 1==0:
        model.eval()
        with torch.no_grad():
            outs, _ = model(data.x, data.edge_index)
            train_acc = accuracy_score(data.y[data.train_mask].cpu(), outs[0][data.train_mask].cpu().data.numpy().argmax(-1))
            val_acc = accuracy_score(data.y[data.val_mask].cpu(), outs[0][data.val_mask].cpu().data.numpy().argmax(-1))
            test_acc = accuracy_score(data.y[data.test_mask].cpu(), outs[0][data.test_mask].cpu().data.numpy().argmax(-1))
            if (epoch+1) % args.log_interval==0:
                print(f"epoch{epoch+1}: train_loss={round(loss.item(), 5)}, train_acc={round(train_acc * 100 ,2)}, val_acc={round(val_acc * 100,2)}, test_acc={round(test_acc * 100,2)}")
        if(val_acc > max_val_acc):
            max_val_acc = val_acc
            best_test_acc = test_acc
            cur_step = 0
            print(f"epoch{epoch+1}: update best result: val_acc={round(val_acc * 100, 2)}, test_acc={round(test_acc * 100,2)}")
        else:
            cur_step +=1
            if(cur_step>args.patience):
                print("early stop...")
                break

print('Test Accuracy: {:.8f}'.format(best_test_acc))
with open(log_path, "a") as f:
    f.write(",".join(map(str, vars(args).values()))+",%.2f\n"%(best_test_acc*100))



