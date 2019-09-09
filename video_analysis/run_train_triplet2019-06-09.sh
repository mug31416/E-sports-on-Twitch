#!/bin/bash
batchSize=4
classQty=2
epochQty=20

# Note the reset_auc flag!!!
python ./spatial_cnn_triplet.py --data_root ~/11775_Project/videos_frames --train_desc ~/11775_Project/label_data/train_spatial_train.npy --class_qty $classQty  --epochs $epochQty --test_desc ~/11775_Project/label_data/train_spatial_valid.npy --resume record_triplet/spatial/model_best.pth.tar --batch_size $batchSize --reset_auc

classQty=2
python ./motion_cnn_triplet.py --data_root ~/11775_Project/videos_flow --train_desc ~/11775_Project/label_data/train_flow_train.npy --class_qty $classQty  --epochs $epochQty --test_desc ~/11775_Project/label_data/train_flow_valid.npy --resume record_triplet/motion/model_best.pth.tar --batch_size $batchSize --reset_auc

