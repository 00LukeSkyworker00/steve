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

# Pretrain dvae on MOVi
for dataset in c d e solid tex ; do
    echo "Start pre-training dvae on $dataset"
    python -u train.py \
        --dataset "movi_$dataset" \
        --data_path $DATA_DIR/movi/$dataset \
        --dvae_dir $OUT_DIR/movi/$dataset/dvae \
        --out_path $OUT_DIR/movi/$dataset \
        --epochs 50 \
        --steps 100000000 \
        --use_dp \
        --wandb_proj_sufix _dvae \
        --dvae_pretrain
done

# Pretrain dvae on CaterTex
echo "Start pre-training dvae on CaterTex"
python -u train.py \
    --dataset "CaterTex" \
    --data_path $DATA_DIR/catertex \
    --dvae_dir $OUT_DIR/catertex/dvae \
    --out_path $OUT_DIR/catertex \
    --epochs 50 \
    --steps 100000000 \
    --use_dp \
    --wandb_proj_sufix _dvae \
    --dvae_pretrain
