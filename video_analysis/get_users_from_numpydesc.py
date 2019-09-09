#!/usr/bin/env python

def get_users(x):
  res = []
  for i in range(len(x)):
    res.append(x[i][1].split()[0].split('/')[-1])
  return res
