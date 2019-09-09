#!/usr/bin/env python
import sys, os, random
file_dir="video_list"

random.seed(0)

with open(file_dir + "/video_esea_2019-0409.txt") as f:
  for line in f:
    line = line.strip()
    if line:
      print(line + "#video_downsample/esea/")

MAX_ALL_QTY=5000

arr = []

with open(file_dir + "/video_all_2019-0409.txt") as f:
  for line in f:
    line = line.strip()
    if line:
      arr.append(line)

random.shuffle(arr)

for line in arr[0:MAX_ALL_QTY]:
  print(line + "#video_downsample/all/")
