#!/bin/bash
lrs=(0.001 0.005) # 2
weight_decays1=(0.1 0.001) #3
weight_decays2=(0.0001 0.001) #3
hidden_sizes=(64 32 128) # 4
dropouts=(0.5 0.2 0.8) #3
gat_dropouts=(0 0.5) #3
gat_heads=(2 8) #3
num_layers=(16 32) # 4
residual_fusions=(add)
# lo_ss_trains=(0.2 0.5) #3
# lo_ss_vals=(0.2 0.5 1) #3
# lo_ss_trains=(0.2 0.5) #3
lo_ss_trains=(0.05) #3
# lo_ss_vals=(0.2 0.5 0.1) #3
for lr in ${lrs[@]}
do
    for weight_decay1 in ${weight_decays1[@]}
    do
        for hidden_size in ${hidden_sizes[@]}
        do
            for dropout in ${dropouts[@]}
            do
                for num_layer in ${num_layers[@]}
                do
                    for lo_ss_train in ${lo_ss_trains[@]}
                    do
                        for residual_fusion in ${residual_fusions[@]}
                        do
                            for gat_head in ${gat_heads[@]}
                            do
                                for gat_dropout in ${gat_dropouts[@]}
                                do
                                    for weight_decay2 in ${weight_decays2[@]}
                                    do
                                        # for lo_ss_val in ${lo_ss_vals[@]}
                                        # do
                                        python run2.py \
                                            --data_name citeseer \
                                            --log_interval 10000 \
                                            --lr $lr \
                                            --split public \
                                            --model_name sdgat \
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
                                            --gat_heads $gat_head
                                        # done
                                    done 
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
