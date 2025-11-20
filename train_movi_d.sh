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

# Train Movi-D
echo 'Start pre-training STEVE on Movi-d'
python -u train.py \
    --dataset movi_d \
    --data_path $DATA_DIR/movi/d \
    --out_path $OUT_DIR/movi/d \
    --num_slots 15 \
    --steps 200000 \
    --use_dp \
