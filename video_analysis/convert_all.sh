#!/bin/bash
SRC_DIR=$1
if [ "$SRC_DIR" = "" ] ; then
  echo "Specify source dir (1st arg)"
  exit 1
fi
DST_DIR=$2
if [ "$DST_DIR" = "" ] ; then
  echo "Specify target dir (2d arg)"
  exit 1
fi
QTY=$3
if [ "$QTY" = "" ] ; then
  echo "Specify # of || conversions (3d arg)"
  exit 1
fi
rm -rf logs
mkdir logs_byhashval
for ((i=0;i<$QTY;++i)) ;  do
  nohup ./convert_videos.py --src_dir "$SRC_DIR" --dst_dir "$DST_DIR"  -d $QTY -m $i 2>&1|tee logs_byhashval/log.$i &
done
