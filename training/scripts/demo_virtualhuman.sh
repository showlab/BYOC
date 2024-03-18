#!/usr/bin/env bash
set -ex
CUDA_VISIBLE_DEVICES=1 \
python3 inference_virtualhuman.py \
  --model virtualhumanlab \
  --crop_size 256 \
  --checkpoints_dir checkpoint/virtual_human_lab_M111 \
  --netE Fc50blendshape \
  --style_dim 512 \
  --name NeuralRenderer \
  --dataset_mode virtualhumanlab \
