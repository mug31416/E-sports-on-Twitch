#!/usr/bin/env python3

import os, sys
import argparse
import pandas as pd

MP4_SUFFIX = '.mp4'

def main(argv):
  parser = argparse.ArgumentParser(description='Generate label file (for training/validation or testing)')

  parser.add_argument('--csv_label_file', type=str,
                      required=True,
                      help = 'CVS file with label info')

  parser.add_argument('--out_label_file', type=str,
                      required=True,
                      help = 'A simple-format output label file')

  parser.add_argument('--proc_video_dir', type=str,
                      required=True,
                      help = 'Directory with processed video files.')

  args = parser.parse_args(argv)
  print(args)

  label_data = pd.read_csv(args.csv_label_file)

  label_map = dict()

  users = label_data['twitch'] 
  labels = label_data['rank_A']
  qty = len(users)

  for i in range(qty):
    label_map[users[i]] = int(labels[i])

  with open(args.out_label_file, 'w') as of:
    for video_fn in os.listdir(args.proc_video_dir):  

      if not video_fn.endswith(MP4_SUFFIX):
        raise Exception('Wrong file format:' + video_fn)

      base_fn = video_fn[0:-len(MP4_SUFFIX)]
      #print(video_fn, base_fn)

      last_underscore = base_fn.rfind('_')

      if last_underscore < 0:
        raise Exception('Wrong file name format (no underscore):' + video_fn)
      username = video_fn[0:last_underscore]
      if not username in label_map:
        print('No label for user:' + username + ' ... ignoring')
        continue

      of.write(base_fn + " " + str(label_map[username])+"\n")
    

if __name__ == '__main__':
  main(sys.argv[1:])
