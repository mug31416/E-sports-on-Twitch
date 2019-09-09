#!/bin/bash -e

TEST_FRAME_QTY=10
#BATCH_SIZE=128
BATCH_SIZE=8
CLASS_QTY=2

dir_date="2019-06-09"

if [ "1" = "1" ] ; then

  if [ -d "record" ] ; then 
    echo "The record directory exists, you need to rename/backup it before running predictions!"
    exit 1
  fi

  ln -s "record_${dir_date}" record

  echo "With pretraining using dir"
  file record

  ./genpred_cnn.py --cnn_type spatial --data_root ~/11775_Project/videos_frames --test_desc ~/11775_Project/label_data/pretrain_spatial_train.npy --test_frame_qty $TEST_FRAME_QTY --batch_size $BATCH_SIZE --class_qty $CLASS_QTY --pred_file ~/11775_Project/preds/pretrain_spatial_train.json


  ./genpred_cnn.py --cnn_type motion --data_root ~/11775_Project/videos_flow --test_desc ~/11775_Project/label_data/pretrain_flow_train.npy --test_frame_qty $TEST_FRAME_QTY --batch_size $BATCH_SIZE --class_qty $CLASS_QTY --pred_file ~/11775_Project/preds/pretrain_motion_train.json


  ./genpred_cnn.py --cnn_type spatial --data_root ~/11775_Project/videos_frames --test_desc ~/11775_Project/label_data/pretrain_spatial_test.npy --test_frame_qty $TEST_FRAME_QTY --batch_size $BATCH_SIZE --class_qty $CLASS_QTY --pred_file ~/11775_Project/preds/pretrain_spatial_test.json


  ./genpred_cnn.py --cnn_type motion --data_root ~/11775_Project/videos_flow --test_desc ~/11775_Project/label_data/pretrain_flow_test.npy --test_frame_qty $TEST_FRAME_QTY --batch_size $BATCH_SIZE --class_qty $CLASS_QTY --pred_file ~/11775_Project/preds/pretrain_motion_test.json


  ./genpred_cnn.py --cnn_type spatial --data_root ~/11775_Project/videos_frames --test_desc ~/11775_Project/label_data/train_spatial_train.npy --test_frame_qty $TEST_FRAME_QTY --batch_size $BATCH_SIZE --class_qty $CLASS_QTY --pred_file ~/11775_Project/preds/train_spatial_train.json


  ./genpred_cnn.py --cnn_type motion --data_root ~/11775_Project/videos_flow --test_desc ~/11775_Project/label_data/train_flow_train.npy --test_frame_qty $TEST_FRAME_QTY --batch_size $BATCH_SIZE --class_qty $CLASS_QTY --pred_file ~/11775_Project/preds/train_motion_train.json

  ./genpred_cnn.py --cnn_type spatial --data_root ~/11775_Project/videos_frames --test_desc ~/11775_Project/label_data/train_spatial_valid.npy --test_frame_qty $TEST_FRAME_QTY --batch_size $BATCH_SIZE --class_qty $CLASS_QTY --pred_file ~/11775_Project/preds/train_spatial_valid.json


  ./genpred_cnn.py --cnn_type motion --data_root ~/11775_Project/videos_flow --test_desc ~/11775_Project/label_data/train_flow_valid.npy --test_frame_qty $TEST_FRAME_QTY --batch_size $BATCH_SIZE --class_qty $CLASS_QTY --pred_file ~/11775_Project/preds/train_motion_valid.json

  rm "record"
fi

if [ "1" = "1" ] ; then

  if [ -d "record" ] ; then 
    echo "The record directory exists, you need to rename/backup it before running predictions!"
    exit 1
  fi

  ln -s "record_${dir_date}_no_pretrain" record

  echo "WithOUT pretraining using dir"
  file record

  ./genpred_cnn.py --cnn_type spatial --data_root ~/11775_Project/videos_frames --test_desc ~/11775_Project/label_data/train_spatial_valid.npy --test_frame_qty $TEST_FRAME_QTY --batch_size $BATCH_SIZE --class_qty $CLASS_QTY --pred_file ~/11775_Project/preds/train_wo_pretrain_spatial_valid.json


  ./genpred_cnn.py --cnn_type motion --data_root ~/11775_Project/videos_flow --test_desc ~/11775_Project/label_data/train_flow_valid.npy --test_frame_qty $TEST_FRAME_QTY --batch_size $BATCH_SIZE --class_qty $CLASS_QTY --pred_file ~/11775_Project/preds/train_wo_pretrain_motion_valid.json


  ./genpred_cnn.py --cnn_type spatial --data_root ~/11775_Project/videos_frames --test_desc ~/11775_Project/label_data/train_spatial_train.npy --test_frame_qty $TEST_FRAME_QTY --batch_size $BATCH_SIZE --class_qty $CLASS_QTY --pred_file ~/11775_Project/preds/train_wo_pretrain_spatial_train.json


  ./genpred_cnn.py --cnn_type motion --data_root ~/11775_Project/videos_flow --test_desc ~/11775_Project/label_data/train_flow_train.npy --test_frame_qty $TEST_FRAME_QTY --batch_size $BATCH_SIZE --class_qty $CLASS_QTY --pred_file ~/11775_Project/preds/train_wo_pretrain_motion_train.json

  rm "record" 
fi
