#!/bin/bash

epoch_qty=10

mkdir -p record/spatial
./spatial_cnn.py --data_root ~/11775_Project/videos_frames --train_desc ~/11775_Project/label_data/pretrain_spatial_train.npy --class_qty 10  --epochs $epoch_qty --test_desc ~/11775_Project/label_data/pretrain_spatial_test.npy --resume record/spatial/model_best.pth.tar --batch_size 4 

mkdir -p record/motion
./motion_cnn.py --data_root ~/11775_Project/videos_flow/ --train_desc ~/11775_Project/label_data/pretrain_flow_train.npy --class_qty 10 --epochs $epoch_qty --test_desc ~/11775_Project/label_data/pretrain_flow_test.npy --resume record/motion/model_best.pth.tar --batch_size 4

