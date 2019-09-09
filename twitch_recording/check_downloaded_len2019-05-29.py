#!/usr/bin/env python
import sys, os, random
import subprocess
file_dir="video_list"

dstPath = sys.argv[1]
allSizeFileName = sys.argv[2]
wrongSizeListFileName = sys.argv[3]

totQty=0
downloadQty=0
errQty=0

dirList = set()

def getFileSize(fn):
  statinfo = os.stat(fn)
  return statinfo.st_size

for fileList in ['video_esea_2019-0409.txt', 'video_all_2019-0409.txt' ] :
  with open(os.path.join(file_dir, fileList)) as f:
    for fileName in f:
      fileName = fileName.strip()
      dirList.add(os.path.dirname(fileName))

totQty = 0
wrongQty = 0

wrongSizeList = []

dirList = list(dirList)
allSizeList = []
for dirName in dirList:
  output = subprocess.check_output(['s3cmd', 'ls', dirName + '/']).decode('utf8')

  for line in output.split('\n'):
    fullFileAddr = line.strip() 
    if not fullFileAddr: 
      continue
    fnStart = fullFileAddr.find('s3://')
    if fnStart < 0:
      print('File start not found in: "' + line + '"')
      os.exit(1)
    lineStart = fullFileAddr[0:fnStart-1].strip()
    fileSize=int(lineStart.split()[2])
    fileName = fullFileAddr[fnStart:]
    fileSuffix = fileName.replace('s3://11775projecttwitchvideostore/', '')
    dstFile = os.path.join(dstPath, fileSuffix)
    dstFileSize = getFileSize(dstFile)
    totQty += 1
    allSizeList.append( (fileSize, fullFileAddr) )
    if fileSize != dstFileSize:
      print('Size mismatch:', fileSize, fileName, '->', dstFileSize, dstFile)
      wrongSizeList.append(fullFileAddr)
      wrongQty += 1

    if totQty % 10 == 0:
      print(totQty, ' files checked')
  

print('Files checked:', totQty, 'Wrong file size:', wrongQty)

with open(allSizeFileName, 'w') as f:
  for sz, fn in allSizeList:  
    f.write(str(sz) + ' ' + fn + '\n')

with open(wrongSizeListFileName, 'w') as f:
  for fn in wrongSizeList:
    f.write(fn + '\n')
