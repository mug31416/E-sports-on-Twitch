import numpy as np
import random


def set_seed(seed):
  print('Set seed: ', seed)
  random.seed(seed)
  np.random.seed(seed)
  return seed

