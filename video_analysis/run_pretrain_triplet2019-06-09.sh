#!/bin/bash -e
batchSize=4
classQty=10
epochQty=10

python ./spatial_cnn_triplet.py --data_root ~/11775_Project/videos_frames --train_desc ~/11775_Project/label_data/pretrain_spatial_train.npy --class_qty $classQty  --epochs $epochQty --test_desc ~/11775_Project/label_data/pretrain_spatial_test.npy --batch_size $batchSize #--reset_auc

python ./motion_cnn_triplet.py --data_root ~/11775_Project/videos_flow --train_desc ~/11775_Project/label_data/pretrain_flow_train.npy --class_qty $classQty  --epochs $epochQty --test_desc ~/11775_Project/label_data/pretrain_flow_test.npy --batch_size $batchSize #--reset_auc
