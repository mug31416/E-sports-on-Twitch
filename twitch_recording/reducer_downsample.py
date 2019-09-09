#!/usr/bin/env python3
import sys, os
import subprocess
import tempfile

VIDEO_DOWNSAMPLE_ROOT = "s3://11775projecttwitchvideostore/"

downsampling_video_len=300
downsampling_frame_rate=10

for line in sys.stdin:

  print(line)
  desc, _ = line.strip().split('\t')

  file_name, dest_subfolder = desc.split('#')

  #print("@@@@@@@@@@@@@@"+filename)


  video_src = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
  video_src_name = video_src.name
  video_src.close()

  video_dst = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
  video_dst_name = video_dst.name
  video_dst.close()


  conv_line = f'ffmpeg -y -ss 0 -i {video_src_name} -strict experimental -t {downsampling_video_len} -r {downsampling_frame_rate} {video_dst_name}'


  # Don't have to quote file name, b/c subprocess does this for us
  subprocess.check_call(['s3cmd', 'get', file_name, video_src.name, '--force'])

  subprocess.check_call(conv_line.split())

  dest_path = os.path.join(VIDEO_DOWNSAMPLE_ROOT, dest_subfolder)

  arr = os.path.basename(file_name).split()
  short_file_name_arr = [arr[0], arr[2], arr[3], '.mp4']

  subprocess.check_call(['s3cmd', 'put', video_dst.name, os.path.join(dest_path, f'{downsampling_video_len}_{downsampling_frame_rate}', ' '.join(short_file_name_arr))])

  os.unlink(video_src_name)
  os.unlink(video_dst_name)


