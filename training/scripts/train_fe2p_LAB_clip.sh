#!/usr/bin/env bash
set -ex

CUDA_VISIBLE_DEVICES=2 \
python3 train.py \
  --dataroot ./datasets/FFHQ_resized \
  --dataset_mode fe2p \
  --model fe2p \
  --netV fan \
  --netE fc50blendshape \
  --netG modulate \
  --checkpoints_dir checkpoint/FE2P \
  --BiSeNet_path checkpoint/pretrained/face_parse_79999_iter.pth \
  --FER_path checkpoint/pretrained/PrivateTest_model.t7 \
  --netE_path checkpoint/virtual_human_lab_M111/NeuralRenderer/latest_net_E.pth \
  --netG_path checkpoint/virtual_human_lab_M111/NeuralRenderer/latest_net_G.pth \
  --no_dropout \
  --no_flip \
  --crop_size 256 \
  --print_freq 20 \
  --save_epoch_freq 20 \
  --display_freq 20 \
  --display_server localhost \
  --display_port 8900 \
  --display_env fe2pClipNoLoop \
  --name fe2pClipNoLoop \
  --batchSize 16 \
  --lr 3e-5 \
  --lambda_Exp 0.1 \
  --lambda_Seg 0.5 \
  --lambda_Loop 0.0 \
  --n_epochs 50 \
  --n_epochs_decay 50 \
  --style_dim 512 \
  --motion_dim 50 \
  --no_TTUR \
  --norm_blendshape clip \

