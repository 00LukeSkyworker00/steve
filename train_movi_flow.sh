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
echo 'Start pre-training STEVE on Movi-flow'
python -u train.py \
    --dataset movi_flow \
    --data_path $DATA_DIR/movi/flow \
    --out_path $OUT_DIR/movi/flow \
    --num_slots 12 \
    --steps 200000 \
    --use_dp \
