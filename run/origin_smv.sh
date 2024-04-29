#!/bin/bash
cd ../
data_name=cora
model_names=(gcn)
num_layers=(16) # 4
hidden_size=64
weight_decay=0.001
loop_time=100
lr=0.01
for model_name in ${model_names[@]}
do
    for num_layer in ${num_layers[@]}
    do
        for ((i=0;i<$loop_time;i++))
        do
            python run_baseline_hs.py \
                --data_name $data_name \
                --log_interval 10000 \
                --lr $lr \
                --split full \
                --model_name $model_name \
                --weight_decay $weight_decay \
                --hidden_size $hidden_size \
                --num_layer $num_layer \
                --seed $i \
                --gin_eps_trainable \
                --log_dir ./log_analysis/smv2
        done 
    done
done