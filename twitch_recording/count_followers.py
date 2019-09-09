#!/usr/bin/env python
import os, sys, requests, json

def get_followers(twitch_id):

  """
  -1 as ERROR Code
  """
  client_id = "rs3o3mnh6gauh2zba8xj9r2po7s5ri"
  r = requests.get(url, headers={"Client-ID": client_id}, timeout=15)
  try:
    r.raise_for_status()
    info = r.json()
    return info["followers"]
  except Exception as e:
    return -1

if __name__ == '__main__':


  if False:
    f = open("esea_streamers.csv")
    users = [line.split(",")[1].strip() for line in f.readlines()[1:]]
    f.close()
  else:
    f = open("../label_data/users_all_only_list.txt")
    users = [line.strip() for line in f]

  client_id = "rs3o3mnh6gauh2zba8xj9r2po7s5ri"
  oauth_token = "n7x192mz4jlm60ney4d68fo9qdg7h0"

  res = {}

  print ("twitch,followers")

  for u in users:
    url = "https://api.twitch.tv/kraken/channels/%s" % u
    r = requests.get(url, headers={"Client-ID": client_id}, timeout=15)
    try:
      r.raise_for_status()
      info = r.json()
      fqty = info["followers"]
      print("%s,%d" %(u, fqty))
      res[u] = { "followers" : fqty }
    except Exception as e:
      print ("%s,-1" % u)
  
  with open('../label_data/num_of_followers_new.json', 'w') as f:
    json.dump(res, f)
