#!/usr/bin/env bash
set -ex
CUDA_VISIBLE_DEVICES=1 \
python3 train.py \
  --dataset_mode exp2bsmix \
  --exp_basis_folder_virtual /home/zechen/Deep3DFaceRecon_pytorch/checkpoints/facerecon/results/M111_combined_cam_front/epoch_20_000000 \
  --exp_basis_folder_real /home/zechen/Deep3DFaceRecon_pytorch/checkpoints/facerecon/results/FFHQ_resized_256/epoch_20_000000 \
  --blendshape_file /home/zechen/Deep3DFaceRecon_pytorch/datasets/M111_combined_blendshape.json \
  --model exp2bsmix \
  --netV exp2bsonelayer \
  --netE fc50blendshape \
  --netG modulate \
  --checkpoints_dir checkpoint/EXP2BSMIX \
  --netE_path checkpoint/virtual_human_ou_M111/NeuralRenderer/latest_net_E.pth \
  --netG_path checkpoint/virtual_human_ou_M111/NeuralRenderer/latest_net_G.pth \
  --no_dropout \
  --no_flip \
  --crop_size 256 \
  --print_freq 20 \
  --save_epoch_freq 1 \
  --display_freq 20 \
  --display_server localhost \
  --display_port 8900 \
  --display_ncols 2 \
  --display_env Exp2BlendshapeMixRegressOneLayer \
  --name Exp2BlendshapeMixRegressOneLayer \
  --batchSize 64 \
  --lr 3e-4 \
  --n_epochs 2 \
  --n_epochs_decay 8 \
  --no_TTUR \
  --style_dim 512 \
  --norm_blendshape none \

