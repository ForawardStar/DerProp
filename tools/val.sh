#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

config=configs/pascal.yaml
checkpoint_path=/home/lmj/fyb/SemiSemSeg8/exp/pascal/92/corrmatch/resnet101_76.603.pth


export LD_PRELOAD=/home/lmj/miniforge3/envs/corrmatch/lib/python3.9/site-packages/torch/lib/libtorch_cpu.so
CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    evaluate.py \
    --config=$config --checkpoint_path $checkpoint_path
