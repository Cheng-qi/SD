#!/bin/bash
lr=0.01
data_name=cora
model_name=sdgat
weight_decay1=0.01
weight_decay2=0.01
hidden_size=128
dropout=0.5
num_layer=16
residual_fusion=add
loss_lambda=0.1
gat_heads=2
gat_dropout=0.5
python main.py \
    --data_name $data_name \
    --log_interval 10000 \
    --lr $lr \
    --model_name $model_name \
    --weight_decay1 $weight_decay1 \
    --weight_decay2 $weight_decay2 \
    --hidden_size $hidden_size \
    --dropout $dropout \
    --residual_fusion $residual_fusion \
    --num_layer $num_layer \
    --loss_lambda $loss_lambda \
    --gat_heads $gat_heads \
    --gat_dropout $gat_dropout \
