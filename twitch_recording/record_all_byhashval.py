import requests
import os, sys
import time, datetime
import argparse
import json
import subprocess

from record_base import TwitchRecorderBase

BATCH_QTY=10
GAME_NAME="Counter-Strike: Global Offensive"

def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('-d', dest='userlist_hash_div', type=int, required=True)
  parser.add_argument('-m', dest='userlist_hash_rem', type=int, required=True)
  parser.add_argument('-q', dest='quality', type=str, default='480p,worst')
  parser.add_argument('-r', dest='refresh', type=float, default=30.)

  return parser.parse_args()

def readNextPlayerBatch(client_id, offset, batch_qty, game_name, timeout=15):
  results = []
    
  next_offset = offset
    
  while len(results) < batch_qty:
    print('Current offset: %d result size: %d' % (next_offset, len(results)))
    uri='https://api.twitch.tv/kraken/streams?offset=%d'  % next_offset
    r = requests.get(uri, headers={"Client-ID": client_id}, timeout=15)
    js = r.json()
    if "streams" not in js:
      print(json.dumps(r.json(), indent=4, sort_keys=False))
      raise Exception("No streams found!")
    streams=js['streams']
    
    qty = len(streams)
    if qty == 0:
      break
    next_offset += qty
      
    for e in streams:
      if e["game"] == game_name and e["stream_type"] == "live":
        results.append(e["channel"]["name"])
          
  return (next_offset, results)


class TwitchRecorderAllUsers(TwitchRecorderBase):
  """
  Record Twitch videos.

  """
  def __init__(self, refresh, userlist_hash_rem, userlist_hash_div, quality):
    TwitchRecorderBase.__init__(self, 
                               "s3://11775projecttwitchvideostore/twitch-allusers-5min-once-a-day/", 
                               refresh, quality, 
                               "recorded_all", "processed_all")

    self.userlist_hash_div = userlist_hash_div
    self.userlist_hash_rem = userlist_hash_rem
    self.userlist = []
    self.ptr = 0
    self.next_offset = 0 # The offset we use to download active streams from twitch

  def use_username(self, username):
    v = hash(username) % self.userlist_hash_div
    return v == self.userlist_hash_rem

  def get_next_user(self):

    while True:
      if self.ptr < len(self.userlist):
        username = self.userlist[self.ptr]
        self.ptr += 1

        sel_cond = self.use_username(username)
        print('Testing username %s does it satisfy the selection criterion? %d' % (username, int(sel_cond)))

        if sel_cond:
          return username
        continue

      print('The temp list is empty, retrieving new batch, offset %d' % self.next_offset)
      self.next_offset, self.userlist = readNextPlayerBatch(self.client_id, self.next_offset, BATCH_QTY, GAME_NAME, timeout=15)
      self.ptr = 0
      print('Retrieved %d entries, next offset is %d' % (len(self.userlist), self.next_offset))

      if len(self.userlist) == 0:
        self.next_offset = 0

      print('Sleeping %g secs' % self.refresh)
      time.sleep(self.refresh)


if __name__ == '__main__':
  args = parse_arguments()
  recorder = TwitchRecorderAllUsers(args.refresh, args.userlist_hash_rem, args.userlist_hash_div, args.quality)
  recorder.run()

