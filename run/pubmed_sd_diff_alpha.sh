#!/bin/bash
cd ../
data_name=pubmed
num_layer=16 # 4
lo_ss_trains=(1.5 2)
loop_time=10
# pubmed-sdgcn
lr=0.01 # 2
weight_decay1=0.01 #3
weight_decay2=0.0001 #3
hidden_size=64 # 4
dropout=0.5  #3
gat_dropout=0.5 #3
gat_head=2 #3
residual_fusion=add
model_name=sdgcn

# pubmed-sdgin
# model_name=sdgin
# lr=0.01 # 2
# weight_decay1=0.001 #3
# weight_decay2=0.001 #3
# hidden_size=64 # 4
# dropout=0.5  #3
# gat_dropout=0.5 #3
# gat_head=2 #3
# residual_fusion=add
# pubmed-sdgat
# model_name=sdgat
# lr=0.01 # 2
# weight_decay1=0.001 #3
# weight_decay2=0.0001 #3
# hidden_size=32 # 4
# dropout=0.2  #3
# gat_dropout=0.5 #3
# gat_head=2 #3
# residual_fusion=add
# lo_ss_train=0.001 #3
for lo_ss_train in ${lo_ss_trains[@]}
do
    for ((i=1;i<loop_time;i++))
    do
        python run2.py \
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
            --log_dir ./log_analysis/diff_alpha
    done 
done
