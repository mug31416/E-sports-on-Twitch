#!/usr/bin/env python3
import sys, os, argparse
import cv2

from pytorch_data import GetFrames, resize_img


def main(argv):
  parser = argparse.ArgumentParser(description='Process videos (cut and subsample)')

  parser.add_argument('--src_dir', type=str,
                      required=True,
                      help = 'Source directory')
  parser.add_argument('--dst_dir', type=str,
                      required=True,
                      help = 'Target directory')
  parser.add_argument('--file_name', type=str,
                      required=True,
                      help = 'Target directory')
  parser.add_argument('--min_width_height', type=int,
                      default=256,
                      help = 'Min width/height')
  parser.add_argument('--jpg_quality', type=int,
                      default=90,
                      help = 'JPG quality')
  parser.add_argument('--max_dur', type=int,
                      default=120,
                      help = 'max duration in seconds')



  args = parser.parse_args(argv)
  print(args)


  file_name = args.file_name

  sub_folder = file_name.replace('.mp4', '').strip()
  dst_dir = os.path.join(args.dst_dir, sub_folder)

  if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

  print(file_name, sub_folder)

  full_inp_path = os.path.join(args.src_dir, file_name)

  if not os.path.exists(full_inp_path):
    print('File does not exist:', full_inp_path)
    sys.exit(1)

  frame_iter = GetFrames(full_inp_path)
  
  while frame_iter.read_next():

    if frame_iter.fps * args.max_dur <= frame_iter.frame:
      break
  
    frame_file_name = os.path.join(dst_dir, 'frame%06d.jpg' % frame_iter.frame)

    if True:
      resized_img, target_size_width, target_size_height  = resize_img(frame_iter.img, frame_iter.width, frame_iter.height, args.min_width_height)
      #print('Image dimensions (%d, %d) -> (%d, %d):' % (frame_iter.width, frame_iter.height, target_size_width, target_size_height))
    else:
      resized_img = frame_iter.img
      args.jpg_quality=100
        
    cv2.imwrite(frame_file_name, resized_img, [cv2.IMWRITE_JPEG_QUALITY, args.jpg_quality])

  
  frame_iter.release()

if __name__ == '__main__':
  main(sys.argv[1:])
