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

DATASET=tex
# Pretrain dvae on MOVi
echo "Start pre-training dvae on $DATASET"
python -u train.py \
    --dataset "movi_$DATASET" \
    --data_path $DATA_DIR/movi/$DATASET \
    --dvae_dir $OUT_DIR/movi/$DATASET/dvae \
    --out_path $OUT_DIR/movi/$DATASET \
    --batch_size 72 \
    --epochs 50 \
    --steps 100000000 \
    --use_dp \
    --wandb_proj_sufix _dvae \
    --dvae_pretrain
