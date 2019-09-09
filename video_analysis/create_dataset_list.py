#!/usr/bin/env python

import argparse
import os, sys
import json
import pandas as pd
import numpy as np

MIN_FRAME_QTY=20

def readLabels(labelFile):

  res = {}
  if labelFile.endswith('.json'):

    with open(labelFile) as f:
      dt = json.load(f)
      for uname, info in dt.items(): 
        res[uname]=info['cluster'] 

  elif labelFile.endswith('.csv'):

    dt = pd.read_csv(labelFile)
    uname = dt['twitch']
    label = dt['rank_A']
    qty = len(uname)
    assert(len(label) == qty)
    for i in range(qty):
      res[uname[i]] = label[i] 

  else:
    raise Exception('Wrong file extension in file: ' + labelFile)

  return res

def getFrameQty(dirName):

  maxQty = 0
  for f in os.listdir(dirName):
    if f.startswith('frame') and f.endswith('.jpg'):
      qty = int(f.replace('frame','').replace('.jpg',''))
      maxQty = max(qty, maxQty)

  return maxQty

def getTwichIdFromDirName(dn):
  if not dn: 
    return ''
  return dn.split()[0]
    
#
# Find data directories
# Here's a bit of hardcoding related to how flow data is stored. 
#
def findDataDirs(rootDir, userLabels):
  res = []  

  for fn in os.listdir(rootDir):
    subDir = os.path.join(rootDir, fn)
    if os.path.isdir(subDir):
      #or fn.startswith('esea'): #FIXME
      if fn == 'y':
        continue
      # Directory names shouldn't be empty
      tid = getTwichIdFromDirName(fn)
      # Leaf directory
      # The extra sub-directory check is extra-slow, so let's not use it
      if tid in userLabels: #and hasFrameFiles(subDir):
        fqty = getFrameQty(subDir)
        if fqty >= MIN_FRAME_QTY:
          res.append( (tid, userLabels[tid], subDir, fqty) )
        print(res[-1])
      # Otherwise dive recursively
      else:
        print('Off the list:', fn)
        res.extend(findDataDirs(subDir, userLabels))

  return res

def main(argv):
  parser = argparse.ArgumentParser(description='Create train/test lists')

  parser.add_argument('--label_file', type=str,
                      metavar='label file name', required=True,
                      help='A label file directory.')
  parser.add_argument('--data_dir', type=str,
                      metavar='data root dir', required=True,
                      help='A directory with images (flow or still RGB)') 
  parser.add_argument('--out_file', type=str,
                      metavar='output numpy file', required=True,
                      help='output numpy file') 

  args = parser.parse_args(argv)

  labelDict = readLabels(args.label_file)

  res_arr = []
  for tid, lab, dn, frameQty in findDataDirs(args.data_dir, labelDict):
    dn = dn.replace("/x/", "/%s/")
    #print(tid, lab, dn)
    res_arr.append(np.array([lab, dn, frameQty]))

  np.save(args.out_file, res_arr)
                      

if __name__ == '__main__':
  main(sys.argv[1:])
