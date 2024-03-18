#!/usr/bin/env bash
set -ex
CUDA_VISIBLE_DEVICES=1 \
python3 train.py \
  --dataroot ./datasets/VirtualHumanLab \
  --model virtualhumanlab \
  --dataset_mode virtualhumanlab \
  --pool_size 50 \
  --character_id M111 \
  --checkpoints_dir checkpoint/virtual_human_lab_M111 \
  --no_dropout \
  --no_flip \
  --print_freq 10 \
  --save_epoch_freq 20 \
  --crop_size 256 \
  --display_freq 10 \
  --display_server localhost \
  --display_port 8900 \
  --name NeuralRenderer \
  --netE Fc50blendshape \
  --batchSize 12 \
  --lr 0.0001 \
  --gan_mode hinge \
  --init_type normal \
  --lambda_D 1.0 \
  --lambda_recon 3.0 \
  --lambda_vgg 10.0 \
  --lambda_feat 10.0 \
  --n_epochs 80 \
  --n_epochs_decay 80 \
  --style_dim 512 \
