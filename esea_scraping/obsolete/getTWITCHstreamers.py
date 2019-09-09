import csv
from bs4 import BeautifulSoup
import re

from os import listdir
from os.path import isfile, join

MYPATH = '/Users/anna/Courses/LargeMultiMedia/project/data/esea/users/'

onlyfiles = [f for f in listdir(MYPATH) if isfile(join(MYPATH, f))]

outf = csv.writer(open('esea_streamers.csv', 'w'))
outf.writerow(['userid', 'twitch'])

i = 0
j = 0
for f in onlyfiles:
  twitchID  = []
  page = BeautifulSoup(open(MYPATH+f), 'html.parser')
  data = page.find_all('a', href=re.compile(r"/twitch.tv/"))
  for d in data:
    twitchID.extend(d.contents)

  print('{}: {}'.format(f, twitchID))

  if len(twitchID)>1:
    print('@@@@@@@@@@@@@@@')
    outf.writerow([f, twitchID[1]])
    j = j + 1
    #exit(0)

  if len(twitchID)==1:
    outf.writerow([f, twitchID[0]])
    j = j + 1

  i = i + 1

print('PLAYERS PROCESSED ', i)
print('PLAYERS WITH TWITCH ', j)


