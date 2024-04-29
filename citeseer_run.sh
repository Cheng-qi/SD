#!/bin/bash
lrs=(0.01 0.002) # 2
weight_decays=(0.01 0.005) #3
hidden_sizes=(128 100 90 256 32) # 4
dropouts=(0.5) #3
num_layers=(2 8 16 32 50) # 4
residual_fusions=(cat add)
# lo_ss_trains=(0.2 0.5) #3
# lo_ss_vals=(0.2 0.5 1) #3
# lo_ss_trains=(0.2 0.5) #3
lo_ss_trains=(0.2 0.5 0.1) #3
lo_ss_vals=(0.2 0.5 0.1) #3
for lr in ${lrs[@]}
do
    for weight_decay in ${weight_decays[@]}
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
                            for lo_ss_val in ${lo_ss_vals[@]}
                            do
                            # for lo_ss_val in ${lo_ss_vals[@]}
                            # do
                            python run.py \
                                --data_name citeseer \
                                --log_interval 10000 \
                                --lr $lr \
                                --weight_decay $weight_decay \
                                --hidden_size $hidden_size \
                                --dropout $dropout \
                                --residual_fusion $residual_fusion \
                                --num_layer $num_layer \
                                --lo_ss_train $lo_ss_train \
                                --lo_ss_val $lo_ss_val \
                                --lo_ss_test $lo_ss_val
                            done
                        done
                    done
                done
            done
        done
    done
done
