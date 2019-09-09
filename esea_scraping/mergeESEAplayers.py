import json
import numpy as np
import os

DATA_PATH = '/Users/anna/Courses/LargeMultiMedia/project/data/esea/'
LEADER = ['Leaderboard_2019-01','Leaderboard_2019-02','Leaderboard_2019-03']

players = dict()

for n in LEADER:

  file_name  = os.path.join(DATA_PATH,n +'_esea_player_links.csv')
  data = np.genfromtxt(fname = file_name, delimiter=',', dtype=str)

  print(n)

  for i in range(data.shape[0]):

    if i == 0:
      continue

    player_id = data[i,3]

    if player_id not in players:
      players[player_id] = dict()

    players[player_id][n] = {'rank_group' : data[i,1], 'position' : data[i,2]}

    print(player_id)
    print(players[player_id])


print('NUMBER OF PLAYERS',len(players))

with open(DATA_PATH+'player_list.json', 'w') as outfile:
  json.dump(players, outfile)
