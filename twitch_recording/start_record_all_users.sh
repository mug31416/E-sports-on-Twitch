#!/bin/bash
rm -rf logs
mkdir logs_byhashval
qty=5
for ((i=0;i<$qty;++i)) ;  do
  nohup ./loop_record_byhashval.sh $i $qty & 
done
