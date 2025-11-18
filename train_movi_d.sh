#!/bin/bash
DATA_DIR=/work/skyworker0/data/uvos
OUT_DIR=/home/skyworker0/workspace/uvos/result

# Set CUDA devices
# export CUDA_VISIBLE_DEVICES=2
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Go to steve repo
echo 'Enter STEVE repo...'
cd ./workspace/uvos/steve/

# Install Moviepy
pip install moviepy

# Train Movi-D
echo 'Start pre-training STEVE on Movi-d'
python -u train.py \
    --data_path $DATA_DIR/movi/d \
    --log_path $OUT_DIR/movi/d \
    --num_slots 15\
    --steps 200000\
    --use_dp\
