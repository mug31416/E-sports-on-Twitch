from urllib.request import urlopen
import os, sys
import time
import numpy as np

np.random.seed(0)

from ConnectionManager import ConnectionManager, CHARSET

DATA_PATH = '/Users/anna/Courses/LargeMultiMedia/project/data/esea/Leaderboard_2019-03/'

SERIES_LEN = 5

downloaded_pages = os.listdir(os.path.join(DATA_PATH))
print(downloaded_pages)

#Populate to-do list
TODO_LIST_URL = []
TODO_LIST_FN = []

for i in ['A','B','C','D']:
  for j in ['1','2','3','4','5','6','7','8','9','10']:
    url =  "https://play.esea.net/index.php?s=stats&d=ranks&rank="+i+"&page="+j
    fn = 'rank_index_'+i+'_'+j
    TODO_LIST_URL.append(url)
    TODO_LIST_FN.append(fn)


cm = ConnectionManager()

# Using solution from https://stackoverflow.com/a/312464
def chunks(arr, n):
  """Yield successive n-sized chunks from arr."""
  for i in range(0, len(arr), n):
    yield arr[i:i + n]


for one_chunk in chunks(list(zip(TODO_LIST_URL, TODO_LIST_FN)), SERIES_LEN):
  one_chunk.insert(0, ('https://play.esea.net/', None))

  cm.new_identity()

  prev_url = None
  cookie = None

  for i in range(len(one_chunk)):
    print(i)

    url, fn = one_chunk[i]

    print(url)

    if i > 0:
      if fn in downloaded_pages:
        print('ALREADY DOWNLOADED. SKIP.')
        continue

    up = cm.request(url, referer=prev_url, cookie=cookie)
    print(up.getheaders())
    print(up.status)

    text = up.read().decode(CHARSET)
    # print(text)

    if i == 0:
      cookie_arr = []
      for name, val in up.getheaders():
        if name.lower() == 'set-cookie':
          end = val.find(';')
          if end > 0:
            cookie_arr.append(val[0:end+1])
      cookie = ' '.join(cookie_arr)
      print('New cookie:', cookie)
    else:

      with open(os.path.join(DATA_PATH,fn), 'w+') as fu:
        fu.write(text)

    prev_url = url

    secs = np.random.randint(0,10,size = 1)
    print('Wait (sec): ',str(secs[0]))
    time.sleep(secs[0])


