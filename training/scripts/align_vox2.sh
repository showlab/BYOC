set -ex
CUDA_VISIBLE_DEVICES=1 \
python scripts/align_68.py \
  --input_folder_path misc/driving_frames/crop_1 \
  --output_folder_path misc/driving_frames/crop_1_aligned \
