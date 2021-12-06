#!/bin/bash

data=cifar10
root=./cifar2png/cifar10png/train
batch_size=100
real_images=1000

CUDA_VISIBLE_DEVICES=0 python3.8 00_data_prep.py \
        --data $data \
        --root $root \
	--batch_size $batch_size\
	--real_images $real_images\


