#!/bin/bash
hv=$1
if [ "$hv" = "" ] ; then
  echo "Usage: <hash remainder> <divisor>"
  exit 1
fi
div=$2
if [ "$div" = "" ] ; then
  echo "Usage: <hash remainder> <divisor>"
  exit 1
fi
if [ ! -d "logs_byhashval" ] ; then
  mkdir "logs_byhashval"
fi
while [ true ] ; do
  python3 -u record_all_byhashval.py -m "$hv" -d "$div" &>logs_byhashval/log.${hv}_${div}
  echo "Python crashed, uups, waiting 10 secs before restarting"
  sleep 10
done
