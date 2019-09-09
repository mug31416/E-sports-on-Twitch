#!/usr/bin/env python

import numpy as np

def numpy_to_list(x):
  if len(x.shape) == 1:
    return list(x)
  return [numpy_to_list(e) for e in x]


print(numpy_to_list(np.arange(0)))
print(numpy_to_list(np.arange(10).reshape(2,5)))
print(numpy_to_list(np.arange(10).reshape(5,2)))
print(numpy_to_list(np.arange(8).reshape(2,2,2)))
  
