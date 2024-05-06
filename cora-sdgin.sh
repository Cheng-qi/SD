#!/bin/bash
lr=0.001
data_name=cora
model_name=sdgin
weight_decay1=0.001
weight_decay2=0.001
hidden_size=128
dropout=0.5
num_layer=16
residual_fusion=add
loss_lambda=0.01

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
    --gin_eps_trainable

