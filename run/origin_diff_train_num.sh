#!/bin/bash
cd ../
data_name=cora
model_names=(gat gin gcn)
per_classes=(10 20 30 40 50 100)
num_layer=2 # 4
hidden_size=64
weight_decay=0.001
loop_time=10
lr=0.01
for model_name in ${model_names[@]}
do
    for per_class in ${per_classes[@]}
    do
    for ((i=0;i<$loop_time;i++))
    do
        python run_baseline.py \
            --data_name $data_name \
            --log_interval 10000 \
            --lr $lr \
            --split random \
            --model_name $model_name \
            --per_class $per_class \
            --weight_decay $weight_decay \
            --hidden_size $hidden_size \
            --num_layer $num_layer \
            --seed $i \
            --gin_eps_trainable \
            --log_dir ./log_analysis/diff_train_num
    done 
    done
done
