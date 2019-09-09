#!/usr/bin/env python
import sys, os, random
import subprocess
file_dir="video_list"

dstPath = sys.argv[1]

totQty=0
downloadQty=0
errQty=0

for fileList in ['video_esea_2019-0409.txt', 'video_all_2019-0409.txt' ] :
  with open(os.path.join(file_dir, fileList)) as f:
    for fileName in f:
      fileName = fileName.strip()
      fileSuffix = fileName.replace('s3://11775projecttwitchvideostore/', '')
      dstFile = os.path.join(dstPath, fileSuffix)
      totQty +=1
      if os.path.exists(dstFile):
        print('Ignoring already downloaded file: ', fileName)
        continue

      try:
        subprocess.check_call(['s3cmd', 'get', fileName, dstFile, '--force'])
        downloadQty +=1
      except:
        print('Error downloading file: ', fileName)
        errQty += 1

  #print(f's3cmd get --recursive {d}')

print('Total # of files:', totQty)
print('Downloaded # of files:', downloadQty)
print('Errored # of files:', errQty)
