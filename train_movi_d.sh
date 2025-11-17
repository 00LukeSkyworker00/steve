#!/bin/bash
DATA_DIR=/work/data/uvos/
OUT_DIR=/home/workspace/uvos/result/

# Set CUDA devices
# export CUDA_VISIBLE_DEVICES=2
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Train Movi-D
python train.py \
    --data_path $DATA_DIR/movi/d \
    --log_path $OUT_DIR/movi/d \
    --num_slots 15\
    --steps 200000\