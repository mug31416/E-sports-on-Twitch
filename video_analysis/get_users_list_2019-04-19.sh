#!/bin/bash -e
# From the videos_flow directory:
cd ../videos_downsample_cut
ls all*|cut -d \  -f 1 |sort -u > ../label_data/users_all_list.txt
ls esea*|cut -d \  -f 1 |sort -u > ../label_data/users_esea_list.txt
