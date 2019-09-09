import json
import os
import time
import numpy as np
from ConnectionManagerJS import ConnectionManagerJS

#NOTES: 1. Tor should be running
#       2. System-wide firewall would need to be set

np.random.seed(0)
DATA_PATH = '/Users/anna/Courses/LargeMultiMedia/project/data/esea/'
PLAYER = 'player_list.json'
USERS = 'users'
SERIES_LEN = 5

# Shutdown actions ---------------------
cm = None

def shutdown():
  if cm is not None: cm.shutdown()

import atexit
atexit.register(shutdown)
#-------------------------------------

with open(DATA_PATH+PLAYER) as f:
  players = json.load(f)

downloaded_pages = os.listdir(os.path.join(DATA_PATH,USERS))
print(downloaded_pages)


# Using solution from https://stackoverflow.com/a/312464
def chunks(arr, n):
  """Yield successive n-sized chunks from arr."""
  for i in range(0, len(arr), n):
    yield arr[i:i + n]


#Populate to-do list
TODO_LIST_URL = []
TODO_LIST_FN = []

for pl in players:
  url =  pl
  fn = url.split('/')[-1]

  if fn in downloaded_pages:
    print('ALREADY DOWNLOADED. SKIP.')
    continue

  TODO_LIST_URL.append(url)
  TODO_LIST_FN.append(fn)


cm = ConnectionManagerJS()

for one_chunk in chunks(list(zip(TODO_LIST_URL, TODO_LIST_FN)), SERIES_LEN):
  one_chunk.insert(0, ('https://play.esea.net/', None))

  cm.new_identity()

  for i in range(len(one_chunk)):

    print(i)

    url, fn = one_chunk[i]

    print(url)

    cm.send_request(url)
    #print(up.getheaders())
    #print(up.status)

    text = cm.extract_html()
    # print(text)

    if i > 0:
      with open(os.path.join(DATA_PATH,fn), 'w+') as fu:
        fu.write(text)

    secs = np.random.randint(0,10,size = 1)
    print('Wait (sec): ',str(secs[0]))
    time.sleep(secs[0])



