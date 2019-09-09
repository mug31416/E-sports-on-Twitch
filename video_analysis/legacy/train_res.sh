#!/bin/bash -e
ne=10
sqty=20
fs=128
k=200
python -u ./train_resnet.py -n $ne --lr 0.00025 --drop_out 0.1 --train_list ../SAMPLE_SPLITS/train_small_simple.txt --valid_list ../SAMPLE_SPLITS/valid_small_simple.txt --cache_file_dir ../SAMPLE_SPLITS/videos_proc_cache  --use_cuda --outmod ../models/ver1 --feat_dim 128 --keyframe_interval $k --file_dir ../SAMPLE_SPLITS/videos_proc.small --subset_qty $sqty --unfreeze_resnet --feat_dim $fs 2>&1|tee log.train_resnet
