#!/usr/bin/env python
# Splits number of followers JSON into train/test sub-JSONs
# A lot is hardcardcoded in this script

from sklearn.model_selection import train_test_split

import sys, json
import numpy as np

CLUSTER_QTY=10
SEED=0

labelDir=sys.argv[1]
fracTest=float(sys.argv[2])

allData = json.load(open(labelDir + '/num_of_followers_2019-04-26.json'))

srcBuckets = [[] for e in range(CLUSTER_QTY)]
minBuckQty = [1e10 for e in range(CLUSTER_QTY)]
maxBuckQty = [0 for e in range(CLUSTER_QTY)]

fqtys = []

for k,v in allData.items():
  qty = v['followers']
  fqtys.append(qty)

pcts = np.quantile(fqtys, np.arange(1 + CLUSTER_QTY) / float(CLUSTER_QTY))

for k,v in allData.items():
  qty = v['followers']
  fc = None
  for t in range(CLUSTER_QTY):
    if qty >= pcts[t] and qty <= pcts[t+1]:
      fc = t
      break
  assert(fc is not None)
  v['cluster'] = fc

for k,v in allData.items():
  cid = v['cluster']
  qty = v['followers']
  srcBuckets[cid].append((k,v))
  minBuckQty[cid] = min(minBuckQty[cid], qty)
  maxBuckQty[cid] = max(maxBuckQty[cid], qty)

rngs = []
for i in range(CLUSTER_QTY):
   rngs.append((minBuckQty[i], maxBuckQty[i]))
rngs.sort()
for i in range(1, CLUSTER_QTY + 1):
  print('Pct range:',  pcts[i-1], pcts[i])
  print('Buck range:',  rngs[i-1][0], rngs[i-1][1])
  

trainData = {}
testData = {}

for cid in range(CLUSTER_QTY):
  arr=srcBuckets[cid]
  qty = len(arr)
  assert(qty > 2)
  testQty = max(1, int(fracTest * qty))
  trainObj, testObj = train_test_split(arr, test_size=testQty, random_state=SEED)

  for k,v in trainObj:
    trainData[k] = v
  for k,v in testObj:
    testData[k] = v

print('Training data size:', len(trainData))
with open(labelDir + '/num_of_followers_train.json', 'w') as f:
  json.dump(trainData, f)
  
print('Testing data size:', len(testData))
with open(labelDir + '/num_of_followers_test.json', 'w') as f:
  json.dump(testData, f)
