#!/bin/bash
fp=$1
if [ "$fp" = "" ] ; then
  echo "Usage: <file name without txt>"
  exit 1
fi
while [ true ] ; do
  python3 -u record_from_list.py -u "$fp.txt" &>logs_fromlist/log.$fp
  echo "Python crashed, uups, waiting 10 secs before restarting"
  sleep 10
done
