set -ex
CUDA_VISIBLE_DEVICES=1 \
python3 train.py \
  --model virtualhuman \
  --dataset_mode virtualhuman0730 \
  --pool_size 50 \
  --hdfs_path hdfs://haruna/home/byte_labcv_default/user/baizechen/virtual_human/render_imgs_0730 \
  --blendshape_path hdfs://haruna/home/byte_labcv_default/user/baizechen/virtual_human/blendshape0730.json \
  --checkpoints_dir hdfs://haruna/home/byte_labcv_default/user/baizechen/virtual_human/CKPT \
  --no_dropout \
  --no_flip \
  --print_freq 10 \
  --save_epoch_freq 20 \
  --crop_size 256 \
  --display_freq 10 \
  --display_server 10.206.74.147 \
  --display_port 13010 \
  --name neural_renderer_local \
  --netE Fc46blendshape \
  --batchSize 4 \
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
