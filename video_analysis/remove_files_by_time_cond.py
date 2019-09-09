#!/usr/bin/env python3
# The script deletes n most recently modified entries from the directory
import os, time, sys

if len(sys.argv) != 6:
  print('Usage: <path> <start yyyy-mm-dd> <end yyyy-mm-dd> <# of last entries to delete> <dry run flag: 0 or 1>')
  sys.exit(1)

def parse_ymd(dt, is_start):
  arr = dt.split('-')
  if len(arr) != 3:
    raise Exception('Wrong date: ' + dt)
  res = [int(s) for s in arr]
  if is_start:
    res.extend([0,0])
  else:
    res.extend([24,60])
  return tuple(res)

dir_name = sys.argv[1]
date_start = parse_ymd(sys.argv[2], True)
date_end = parse_ymd(sys.argv[3], False)
qty = int(sys.argv[4])
dry_run = int(sys.argv[5]) == 1

file_list = os.listdir(dir_name)

lst1 = []

for fn in file_list:
  mod_time = os.path.getctime(fn)
  mod_time_parsed = tuple(time.localtime(mod_time))

  if mod_time_parsed >= date_start and mod_time_parsed <= date_end: 
    lst1.append( (mod_time, mod_time_parsed, fn) )

lst1.sort()

lst1 = lst1[-qty:len(lst1)]

print('Files to delete:')
for mod_time, mod_time_parsed, fn in lst1:
  print(mod_time_parsed, fn)

if not dry_run:
  for _, _, fn in lst1:
    os.unlink(fn)
    print('Deleted', fn)
  print(mod_time_parsed, fn)
