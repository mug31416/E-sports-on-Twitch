#!/bin/bash

#mkdir -p record/spatial
#./spatial_cnn.py --data_root ~/11775_Project/videos_frames --train_desc ~/11775_Project/label_data/train_spatial_train.npy --class_qty 10  --epoch 40 --test_desc ~/11775_Project/label_data/train_spatial_valid.npy --resume record/spatial/model_best.pth.tar --batch_size 4  --class_qty_new 2

./motion_cnn.py --data_root ~/11775_Project/videos_flow/ --train_desc ~/11775_Project/label_data/train_flow_train.npy --class_qty 10 --epochs 40 --test_desc ~/11775_Project/label_data/train_flow_valid.npy --resume record/motion/model_best.pth.tar --batch_size 4 --class_qty_new 2

