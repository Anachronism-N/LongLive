#!/bin/bash
# LongLive I2V Training Script
# Run on GPUs 4,5,6,7

export CUDA_VISIBLE_DEVICES=4,5,6,7

# Project path and config
CONFIG=configs/longlive_train_i2v_local.yaml
LOGDIR=/commondocument/group2/LongLive/outputs/longlive_i2v_train
WANDB_SAVE_DIR=/commondocument/group2/LongLive/outputs/wandb

mkdir -p $LOGDIR
mkdir -p $WANDB_SAVE_DIR

echo "CONFIG=$CONFIG"
echo "LOGDIR=$LOGDIR"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

/commondocument/group2/miniconda3/envs/self_forcing/bin/torchrun \
  --nproc_per_node=4 \
  train.py \
  --config_path $CONFIG \
  --logdir $LOGDIR \
  --wandb-save-dir $WANDB_SAVE_DIR \
  --disable-wandb
