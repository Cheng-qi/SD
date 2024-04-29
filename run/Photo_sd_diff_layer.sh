#!/bin/bash
cd ../
data_name=Photo
num_layers=(2 8 16 32) # 4
loop_time=10
# Phot-sdgcn
# lr=0.001 # 2
# weight_decay1=0.001 #3
# weight_decay2=0.0001 #3
# hidden_size=64 # 4
# dropout=0.5  #3
# gat_dropout=0.5 #3
# gat_head=2 #3
# residual_fusion=add
# model_name=sdgcn
# lo_ss_train=0.1 #3
# Photo-sdgin
# model_name=sdgin
# lr=0.001 # 2
# weight_decay1=0.001 #3
# weight_decay2=0.0001 #3
# hidden_size=64 # 4
# dropout=0.5  #3
# gat_dropout=0.5 #3
# gat_head=2 #3
# residual_fusion=add
# lo_ss_train=0.01 #3
# Photo-sdgat
model_name=sdgat
lr=0.001 # 2
weight_decay1=0.00001 #3
weight_decay2=0.00001 #3
hidden_size=64 # 4
dropout=0.2  #3
gat_dropout=0.5 #3
gat_head=1 #3
residual_fusion=add
lo_ss_train=0.1 #3
for num_layer in ${num_layers[@]}
do
    for ((i=1;i<loop_time;i++))
    do
        python run_big.py \
            --data_name $data_name \
            --log_interval 10000 \
            --lr $lr \
            --split full \
            --model_name $model_name \
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
            --log_dir ./log_analysis/diff_layer
    done 
done
