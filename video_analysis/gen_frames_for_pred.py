#!/usr/bin/env python3
import os, pdb, sys, time, datetime
import numpy as np
import torch
import torch.nn as nn
from pytorch_data import *
import argparse

def main(args):

  VideoFramesTest(label_file=args.item_list, file_dir=args.file_dir, 
                  frame_qty=args.frame_qty, keyframe_skip=args.keyframe_skip, keyframe_interval=args.keyframe_interval,
                  cache_file_prefix=args.frame_file_dir)

try:
  if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate frames for test/validation/prediction")

    parser.add_argument('--item_list', type=str, required=True,
                        help='A list of train/test/val items and their labels')

    parser.add_argument('--frame_qty', type=int,
                        default=60,
                        help='A number of frames (per video) to collect')

    parser.add_argument('--keyframe_interval', type=int,
                        default=10,
                        help='Key frame interval')

    parser.add_argument('--keyframe_skip', type=int,
                        default=0,
                        help='A number of starting frames to skip')

    parser.add_argument('--file_dir', type=str,
                        required=True,
                        help='Directory with video files')

    parser.add_argument('--frame_file_dir', type=str,
                        default=None,
                        help='Frame file directory')

    args = parser.parse_args()

    main(args)

except:
  # tb is traceback
  exType, value, tb = sys.exc_info()
  print(value)
  print(tb)
  pdb.post_mortem(tb)
