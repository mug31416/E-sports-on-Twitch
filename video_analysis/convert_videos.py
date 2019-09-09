#!/usr/bin/env python3

import os, sys, subprocess
import argparse

def get_next_qty(qty_dict_ref, key):
  if not key in qty_dict_ref:
    qty_dict_ref[key] =0
    return 0
  else:
    qty_dict_ref[key] += 1
    return qty_dict_ref[key]
  

def main(argv):
  parser = argparse.ArgumentParser(description='Process videos (cut and subsample)')

  parser.add_argument('-d', 
                      dest='userlist_hash_div', 
                      type=int, required=True)
  parser.add_argument('-m', 
                      dest='userlist_hash_rem', 
                      type=int, required=True)
  parser.add_argument('--video_len', type=int,
                      default = '300',
                      help = 'Maximum video duration in seconds')
  parser.add_argument('--frame_rate', type=int,
                      default = '10',
                      help = 'Subsampling frame-rate')
  parser.add_argument('--src_dir', type=str,
                      required=True,
                      help = 'Source directory')
  parser.add_argument('--dst_dir', type=str,
                      required=True,
                      help = 'Target directory')

  args = parser.parse_args(argv)
  print(args)

  user_file_qtys = dict()

  if not os.path.exists(args.dst_dir):
    os.makedirs(args.dst_dir)

  for root, dirs, files in os.walk(args.src_dir):
    for name in files:
        if name.endswith(".mp4"):
          full_path_src = os.path.join(root, name)
          user_name = name.split()[0]


          v = hash(user_name) % args.userlist_hash_div
          use_cond = v == args.userlist_hash_rem

          print(f'User {user_name} process? {use_cond}')

          if use_cond:
            key = get_next_qty(user_file_qtys, user_name)
            dst_name = f"{user_name}_{key}.mp4"
            full_path_dst = os.path.join(args.dst_dir, dst_name)
            print(full_path_src, full_path_dst)

            if os.path.exists(full_path_dst):
              print(f'Ignoring already converted file {full_path_src} -> {full_path_dst}')
            else:
              conv_line = ['ffmpeg', '-y', '-ss',  '0', '-i', f'{full_path_src}',  '-strict',  'experimental', 
                        '-t', str(args.video_len), '-r', str(args.frame_rate), full_path_dst]

              print(conv_line)
              subprocess.check_call(conv_line)
          


if __name__ == '__main__':
  main(sys.argv[1:])

