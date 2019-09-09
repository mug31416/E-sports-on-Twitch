import requests
import os, sys
import time, datetime
import argparse
import json
import subprocess

from record_base import TwitchRecorderBase

def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('-u', dest='userlist', type=str, default='userlist1.txt')
  parser.add_argument('-q', dest='quality', type=str, default='480p,worst')
  parser.add_argument('-r', dest='refresh', type=float, default=15.)

  return parser.parse_args()

class TwitchRecorderFromList(TwitchRecorderBase):
  """
  Record Twitch videos.

  """
  def __init__(self, refresh, userlist, quality):
    TwitchRecorderBase.__init__(self, 
                               "s3://11775projecttwitchvideostore/twitch-5min-once-a-day/", 
                               refresh, quality, 
                               "recorded_fromlist", "processed_fromlist")

    f = open(userlist, "r")
    self.userlist = [l.strip() for l in f.readlines()]
    f.close()
    self.ptr = 0
    self.n_users = len(self.userlist)

  def get_next_user(self):
    username = self.userlist[self.ptr]
    self.ptr = (self.ptr + 1) % self.n_users
    return username


if __name__ == '__main__':
  args = parse_arguments()
  recorder = TwitchRecorderFromList(args.refresh, args.userlist, args.quality)
  recorder.run()

