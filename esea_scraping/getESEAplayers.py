import csv
from bs4 import BeautifulSoup
import os


DATA_PATH = '/Users/anna/Courses/LargeMultiMedia/project/data/esea/'
LEADER_MTH = 'Leaderboard_2019-03'


def fileName(r,p,leader):
  if leader in ['Leaderboard_2019-01','Leaderboard_2019-02']:
    return "index.php?s=stats&d=ranks&rank="+r+"&page="+str(p+1)
  else:
    return "rank_index_"+r+"_"+str(p+1)


userLinks = []
userRankGrp = []
userRankPos = []

ranks = ['A','B','C','D']

f = csv.writer(open(os.path.join(DATA_PATH,LEADER_MTH +'_esea_player_links.csv'), 'w+'))
f.writerow(['leaderboard','rankGrp', 'rankPos','link'])


for r in ranks:

  for p in range(10):

    page = BeautifulSoup(open(os.path.join(DATA_PATH,LEADER_MTH,fileName(r,p,LEADER_MTH))), 'html.parser')
    table = page.find(class_='leaderboard-table')
    users = table.find_all('a')

    for u in range(len(users)):

      print(r,str(p),u)

      userRankGrp.append(r)
      userRankPos.append(u+1)
      lnk = 'https://play.esea.net'+users[u].attrs.get('href')
      userLinks.append(lnk)
      f.writerow([LEADER_MTH, r, p * 50 + u+1, lnk])

