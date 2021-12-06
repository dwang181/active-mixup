#!/bin/bash

#lr=0.05
lr=0.01

data=cifar10
root=./cifar10_data
model=vgg

model_out=./active_student_models/${data}_${model}_student_model_1000

echo "model_out: " ${model_out}
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3.8 ./active_train.py \
                        --lr ${lr} \
                        --data ${data} \
                        --model ${model} \
                        --root ${root} \
                        --model_out ${model_out}.pth \
                        2>&1 | tee -a ./logs/${data}_${model}_student_model_1000.log

