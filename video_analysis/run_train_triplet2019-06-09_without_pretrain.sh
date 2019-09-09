#!/bin/bash -e
batchSize=4
classQty=2
epochQty=10

rm -rf record_triplet

# Note the reset_auc flag!!!
mkdir -p record_triplet/spatial
python ./spatial_cnn_triplet.py --data_root ~/11775_Project/videos_frames --train_desc ~/11775_Project/label_data/train_spatial_train.npy --class_qty $classQty  --epochs $epochQty --test_desc ~/11775_Project/label_data/train_spatial_valid.npy --batch_size $batchSize 

classQty=2
mkdir -p record_triplet/motion
python ./motion_cnn_triplet.py --data_root ~/11775_Project/videos_flow --train_desc ~/11775_Project/label_data/train_flow_train.npy --class_qty $classQty  --epochs $epochQty --test_desc ~/11775_Project/label_data/train_flow_valid.npy --batch_size $batchSize 

