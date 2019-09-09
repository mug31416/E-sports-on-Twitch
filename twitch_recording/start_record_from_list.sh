#!/bin/bash
rm -rf logs_fromlist
mkdir logs_fromlist
for f in $(ls |grep userlist.*txt) ;  do
  fp=$(basename $f .txt)
  echo Starting $f
  nohup ./loop_record_from_list.sh $fp & 
done
