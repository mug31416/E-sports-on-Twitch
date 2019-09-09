#!/bin/bash

TEST_FRAME_QTY=10
#BATCH_SIZE=128
BATCH_SIZE=32
CLASS_QTY=2

#./genpred_cnn.py --cnn_type spatial --data_root ~/11775_Project/videos_frames --test_desc ~/11775_Project/label_data/train_spatial_valid.npy --test_frame_qty $TEST_FRAME_QTY --batch_size $BATCH_SIZE --class_qty $CLASS_QTY --pred_file ~/11775_Project/preds/train_triplet_wo_pretrain_spatial_valid.json --with_svm

#./genpred_cnn.py --cnn_type spatial --data_root ~/11775_Project/videos_frames --test_desc ~/11775_Project/label_data/train_spatial_train.npy --test_frame_qty $TEST_FRAME_QTY --batch_size $BATCH_SIZE --class_qty $CLASS_QTY --pred_file ~/11775_Project/preds/train_triplet_wo_pretrain_spatial_train.json --with_svm


./genpred_cnn.py --cnn_type motion --data_root ~/11775_Project/videos_flow --test_desc ~/11775_Project/label_data/train_flow_train.npy --test_frame_qty $TEST_FRAME_QTY --batch_size $BATCH_SIZE --class_qty $CLASS_QTY --pred_file ~/11775_Project/preds/train_triplet_wo_pretrain_motion_train.json --with_svm

./genpred_cnn.py --cnn_type motion --data_root ~/11775_Project/videos_flow --test_desc ~/11775_Project/label_data/train_flow_valid.npy --test_frame_qty $TEST_FRAME_QTY --batch_size $BATCH_SIZE --class_qty $CLASS_QTY --pred_file ~/11775_Project/preds/train_triplet_wo_pretrain_motion_valid.json --with_svm

