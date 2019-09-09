#!/usr/bin/env bash

# xentropy_with_pretrain
python -u merge_data.py --datapath raw_predictions/xentropy_with_pretrain --audio proba_audio_w_pretrain_wo_triplet_new.txt --chat_txt_trn chat_train.csv --chat_tmp_trn chat_train_temporal.csv --chat_txt_val chat_val.csv --chat_tmp_val chat_val_temporal.csv --img_trn train_spatial_train.json --img_val train_spatial_valid.json --mov_trn train_motion_train.json --mov_val train_motion_valid.json --output data_xentropy_with_pretrain.json 2>&1|tee log_xentropy_with_pretrain

#triplet_with_pretrain
python -u merge_data.py --datapath raw_predictions/triplet_with_pretrain --audio proba_audio_w_pretrain_w_triplet.txt --chat_txt_trn chat_train_triplet.csv --chat_tmp_trn chat_train_triplet_temporal.csv --chat_txt_val chat_val_triplet.csv --chat_tmp_val chat_val_triplet_temporal.csv --img_trn train_triplet_pretrain_spatial_train.json --img_val train_triplet_pretrain_spatial_valid.json --mov_trn train_triplet_pretrain_motion_train.json --mov_val train_triplet_pretrain_motion_valid.json --output data_triplet_with_pretrain.json 2>&1|tee log_triplet_with_pretrain


#xentropy_wo_pretrain
python -u merge_data.py --datapath raw_predictions/xentropy_wo_pretrain --audio proba_audio_wo_pretrain_wo_triplet.txt --chat_txt_trn chat_train_nopretrain.csv --chat_tmp_trn chat_train_nopretrain_temporal.csv --chat_txt_val chat_val_nopretrain.csv --chat_tmp_val chat_val_nopretrain_temporal.csv --img_trn train_wo_pretrain_spatial_train.json --img_val train_wo_pretrain_spatial_valid.json --mov_trn train_wo_pretrain_motion_train.json --mov_val train_wo_pretrain_motion_valid.json --output data_xentropy_wo_pretrain.json 2>&1|tee log_xentropy_wo_pretrain


#triplet_wo_pretrain
python -u merge_data.py --datapath raw_predictions/triplet_wo_pretrain --audio proba_audio_wo_pretrain_w_triplet.txt --chat_txt_trn chat_train_nopretrain_triplet.csv --chat_tmp_trn chat_train_nopretrain_triplet_temporal.csv --chat_txt_val chat_val_nopretrain_triplet.csv --chat_tmp_val chat_val_nopretrain_triplet_temporal.csv --img_trn train_triplet_wo_pretrain_spatial_train.json --img_val train_triplet_wo_pretrain_spatial_valid.json --mov_trn train_triplet_wo_pretrain_motion_train.json --mov_val train_triplet_wo_pretrain_motion_valid.json --output data_triplet_wo_pretrain.json 2>&1|tee log_triplet_wo_pretrain


#best_models
python -u merge_data.py --datapath raw_predictions/best_models --audio proba_audio_w_pretrain_w_triplet.txt --chat_txt_trn chat_train_triplet.csv --chat_tmp_trn chat_train_triplet_temporal.csv --chat_txt_val chat_val_triplet.csv --chat_tmp_val chat_val_triplet_temporal.csv --img_trn train_triplet_wo_pretrain_spatial_train.json --img_val train_triplet_wo_pretrain_spatial_valid.json --mov_trn train_triplet_wo_pretrain_motion_train.json --mov_val train_triplet_wo_pretrain_motion_valid.json --output data_best_models.json 2>&1|tee log_best_models

#poster result reproduction
#python -u merge_data.py --datapath raw_predictions/poster_result --audio proba_audio.txt --chat_txt_trn chat_train.csv --chat_tmp_trn chat_train_temporal.csv --chat_txt_val chat_val.csv --chat_tmp_val chat_val_temporal.csv --img_trn train_spatial_train.json --img_val train_spatial_valid.json --mov_trn train_motion_train.json --mov_val train_motion_valid.json --output data_poster_result.json 2>&1|tee log_poster_result

