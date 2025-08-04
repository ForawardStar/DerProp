#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

config=configs/pascal.yaml
labeled_id_path=/home/lmj/fyb/CorrMatch-main/partitions/pascal/92/labeled.txt
unlabeled_id_path=/home/lmj/fyb/CorrMatch-main/partitions/pascal/92/unlabeled.txt
save_path=exp/pascal/92/DerProp
#config=configs/cityscapes.yaml
#labeled_id_path=partitions/cityscapes/1_4/labeled.txt
#unlabeled_id_path=partitions/cityscapes/1_4/unlabeled.txt
#save_path=exp/cityscapes/1_4/corrmatch

mkdir -p $save_path

export LD_PRELOAD=/home/lmj/miniforge3/envs/corrmatch/lib/python3.9/site-packages/torch/lib/libtorch_cpu.so

CUDA_VISIBLE_DEVICES=3,4 python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    main.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $2 2>&1 | tee $save_path/$now.txt
