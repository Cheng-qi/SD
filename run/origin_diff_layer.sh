#!/bin/bash
cd ../
data_name=Photo
model_names=(gat gin gcn)
num_layers=(2 8 16 32) # 4
hidden_size=256
weight_decay=0.00001
loop_time=10
lr=0.001
for model_name in ${model_names[@]}
do
    for num_layer in ${num_layers[@]}
    do
    for ((i=1;i<$loop_time;i++))
    do
        python run_baseline.py \
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
            --log_dir ./log_analysis/diff_layer2
    done 
    done
done
