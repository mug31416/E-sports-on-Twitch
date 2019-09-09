#!/usr/bin/env python

import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score

class_prob = 0.4

N=10000

for draw_prob in np.arange(10)/10.0:

  y = np.random.binomial(1, class_prob, N)
  p = np.random.binomial(1, draw_prob ,N)

  print("f-scores %g %g" % (f1_score(y, p, pos_label=0), f1_score(y, p, pos_label=1)))

