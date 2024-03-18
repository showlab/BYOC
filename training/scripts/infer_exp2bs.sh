#!/usr/bin/env bash
set -ex
CUDA_VISIBLE_DEVICES=1 \
python3 inference_exp2bs.py \
  --dataset_mode exp2bsmix \
  --exp_basis_folder_virtual /home/zechen/Deep3DFaceRecon_pytorch/checkpoints/facerecon/results/M111_combined_cam_front/epoch_20_000000 \
  --exp_basis_folder_real /home/zechen/Deep3DFaceRecon_pytorch/checkpoints/facerecon/results/FFHQ_resized_256/epoch_20_000000 \
  --blendshape_file /home/zechen/Deep3DFaceRecon_pytorch/datasets/M111_combined_blendshape.json \
  --model exp2bsmix \
  --netV exp2bs \
  --netE fc50blendshape \
  --netG modulate \
  --checkpoints_dir checkpoint/EXP2BSMIX \
  --netE_path checkpoint/virtual_human_ou_M111/NeuralRenderer/latest_net_E.pth \
  --netG_path checkpoint/virtual_human_ou_M111/NeuralRenderer/latest_net_G.pth \
  --crop_size 256 \
  --name Exp2BlendshapeMixRegress \
  --style_dim 512 \
  --norm_blendshape none \
