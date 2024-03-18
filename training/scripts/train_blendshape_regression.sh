#!/usr/bin/env bash
set -ex
CUDA_VISIBLE_DEVICES=0 \
python3 train.py \
  --dataset_mode mix \
  --dataroot ./datasets/VirtualHumanLab \
  --dataroot_real ./datasets/FFHQ_resized \
  --model bsregress \
  --netV fan \
  --netE fc50blendshape \
  --netG modulate \
  --checkpoints_dir checkpoint/FE2P \
  --netE_path checkpoint/virtual_human_lab_M111/NeuralRenderer/latest_net_E.pth \
  --netG_path checkpoint/virtual_human_lab_M111/NeuralRenderer/latest_net_G.pth \
  --no_dropout \
  --no_flip \
  --crop_size 256 \
  --print_freq 20 \
  --save_epoch_freq 2 \
  --display_freq 20 \
  --display_server localhost \
  --display_port 8900 \
  --display_ncols 2 \
  --display_env BlendshapeRegressMix \
  --name BlendshapeRegressMix \
  --batchSize 16 \
  --lr 5e-5 \
  --n_epochs 5 \
  --n_epochs_decay 30 \
  --no_TTUR \
  --motion_dim 50 \
  --style_dim 512 \
  --norm_blendshape none \

