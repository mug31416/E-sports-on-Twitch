#!/bin/bash -e
#cd ../videos_frames

#python -u ../misc_scripts/create_dataset_list.py --label_file ../label_data/num_of_followers_train.json --data_dir . --out_file ../label_data/pretrain_spatial_train.npy
#python -u ../misc_scripts/create_dataset_list.py --label_file ../label_data/num_of_followers_test.json --data_dir . --out_file ../label_data/pretrain_spatial_test.npy

#cd ../videos_flow
#python -u ../misc_scripts/create_dataset_list.py --label_file ../label_data/num_of_followers_train.json --data_dir . --out_file ../label_data/pretrain_flow_train.npy
#python -u ../misc_scripts/create_dataset_list.py --label_file ../label_data/num_of_followers_test.json --data_dir . --out_file ../label_data/pretrain_flow_test.npy


cd ../videos_flow
python -u ../misc_scripts/create_dataset_list.py --label_file ../label_data/train_small.csv --data_dir . --out_file ../label_data/train_flow_train.npy
python -u ../misc_scripts/create_dataset_list.py --label_file ../label_data/valid_small.csv --data_dir . --out_file ../label_data/train_flow_valid.npy

cd ../videos_frames
python -u ../misc_scripts/create_dataset_list.py --label_file ../label_data/train_small.csv --data_dir . --out_file ../label_data/train_spatial_train.npy
python -u ../misc_scripts/create_dataset_list.py --label_file ../label_data/valid_small.csv --data_dir . --out_file ../label_data/train_spatial_valid.npy
