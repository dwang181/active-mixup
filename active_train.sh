#!/bin/bash

#lr=0.05
lr=0.01

data=cifar10
root=./cifar10_data
model=vgg

model_out=./active_student_models/${data}_${model}_student_model_21000

echo "model_out: " ${model_out}
CUDA_VISIBLE_DEVICES=1,0,2,3 python3.8 ./active_train.py \
                        --lr ${lr} \
                        --data ${data} \
                        --model ${model} \
                        --root ${root} \
                        --model_out ${model_out}.pth \
                        2>&1 | tee -a ./logs/${data}_${model}_student_model_21000.log

