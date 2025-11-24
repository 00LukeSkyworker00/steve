#!/bin/bash
DATA_DIR=/work/skyworker0/data/uvos
OUT_DIR=/home/skyworker0/workspace/uvos/result

# Install packages
pip install moviepy
pip install wandb
pip install --upgrade pydantic==2.7.0

# Setup WanDB (put your api key in a txt file in root.)
export WANDB_API_KEY=$(cat .wandb.txt)

# Go to steve repo
echo 'Enter STEVE repo...'
cd ./workspace/uvos/steve/

# Train
echo 'Start pre-training STEVE on Movi-c'
python -u train.py \
    --dataset movi_c \
    --data_path $DATA_DIR/movi/c \
    --out_path $OUT_DIR/movi/c \
    --num_slots 15 \
    --steps 200000 \
    --use_dp \
