import json
import os
import time
import numpy as np
import csv

np.random.seed(0)

DATA_PATH = '/Users/anna/Courses/LargeMultiMedia/project/data/esea/'
PLAYER = 'player_list.json'
USERS = 'users'

LEAD = ['Leaderboard_2019-01','Leaderboard_2019-02','Leaderboard_2019-03']

with open(DATA_PATH+PLAYER) as f:
  players = json.load(f)

outf = csv.writer(open('esea_stats.csv', 'w'))
outf.writerow(['userid', 'Leaderboard_2019-01_rank', 'Leaderboard_2019-01_position',
               'Leaderboard_2019-02_rank', 'Leaderboard_2019-02_position',
               'Leaderboard_2019-03_rank', 'Leaderboard_2019-03_position'])

for key, contents in players.items():

  userid = key.split('/')[-1]

  r1 = None
  p1 = None
  r2 = None
  p2 = None
  r3 = None
  p3 = None

  if LEAD[0] in contents:
    r1 = contents[LEAD[0]]['rank_group']
    p1 = contents[LEAD[0]]['position']

  if LEAD[1] in contents:
    r2 = contents[LEAD[1]]['rank_group']
    p2 = contents[LEAD[1]]['position']

  if LEAD[2] in contents:
    r3 = contents[LEAD[2]]['rank_group']
    p3 = contents[LEAD[2]]['position']


  outf.writerow([userid, r1, p1, r2, p2, r3, p3])
