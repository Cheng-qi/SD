#!/bin/bash
cd ../
data_name=cora
per_classes=(10 20 30 40 50 100) # 4
num_layer=16
loop_time=10
# cora-sdgcn
lr=0.01 # 2
weight_decay1=0.001 #3
weight_decay2=0.001 #3
hidden_size=128 # 4
dropout=0.2  #3
gat_dropout=0.5 #3
gat_head=2 #3
residual_fusion=add
model_name=sdgcn
lo_ss_train=0.001 #3
# Cora-sdgin
# model_name=sdgin
# lr=0.01 # 2
# weight_decay1=0.01 #3
# weight_decay2=0.001 #3
# hidden_size=64 # 4
# dropout=0.5  #3
# gat_dropout=0.5 #3
# gat_head=2 #3
# residual_fusion=add
# lo_ss_train=0.1 #3
# cora-sdgat
# model_name=sdgat
# lr=0.01 # 2
# weight_decay1=0.01 #3
# weight_decay2=0.001 #3
# hidden_size=64 # 4
# dropout=0.5  #3
# gat_dropout=0 #3
# gat_head=2 #3
# residual_fusion=add
# lo_ss_train=0.01 #3
for per_class in ${per_classes[@]}
do
    for ((i=0;i<loop_time;i++))
    do
        python run2.py \
            --data_name $data_name \
            --log_interval 10000 \
            --lr $lr \
            --split random \
            --model_name $model_name \
            --per_class $per_class \
            --weight_decay1 $weight_decay1 \
            --weight_decay2 $weight_decay2 \
            --hidden_size $hidden_size \
            --dropout $dropout \
            --residual_fusion $residual_fusion \
            --num_layer $num_layer \
            --lo_ss_train $lo_ss_train \
            --lo_ss_val $lo_ss_train \
            --lo_ss_test $lo_ss_train \
            --gat_dropout $gat_dropout \
            --gat_heads $gat_head \
            --seed $i \
            --gin_eps_trainable \
            --log_dir ./log_analysis/diff_train_num
    done 
done