import requests
import os, sys
import time, datetime
import argparse
import json
import subprocess

class TwitchRecorderBase:
  """
  Record Twitch videos base class.

  """
  def __init__(self, s3_path, 
              refresh, quality, 
              recorded_subdir, processed_subdir):

    ## HARD CODE METADATA ##
    # self.client_id = "jzkbprff40iqj646a697cyrvl0zt2m6"
    self.client_id = "rs3o3mnh6gauh2zba8xj9r2po7s5ri"
    self.oauth_token = "n7x192mz4jlm60ney4d68fo9qdg7h0"
    self.ffmpeg_path = "ffmpeg"
    self.root_path = "twitch_videos/"
    self.DURATION_LIMIT = "00:05:00"
    ###

    self.recorded_path = os.path.join(self.root_path, recorded_subdir)
    self.processed_path = os.path.join(self.root_path, processed_subdir)
    self.quality = quality
    self.S3_PATH = s3_path

    self.refresh = refresh
    self.last_record_date = dict()

  def get_next_user(self):
    raise NotImplementedError

  def check_user(self):
    """
    0: online,
    1: offline,
    2: not found,
    3: unknown error
    """

    username = self.get_next_user()

    url = "https://api.twitch.tv/kraken/streams/%s" % username
    info = None
    status = 3

    try:
      r = requests.get(url, headers={"Client-ID": self.client_id}, timeout=15)
      r.raise_for_status()
      info = r.json()
      if info["stream"] is None:
        status = 1
      else:
        status = 0
    except requests.exceptions.RequestException as e:
      if e.response and e.response.reason in ["Not Found", "Unprocessable Entity"]:
        status = 2

    return username, status, info


  def loopcheck(self):
    while True:
      username, status, info = self.check_user()

      if status == 2:
        print ("User %s not found." % username)
        time.sleep(self.refresh)
      elif status == 3:
        print ("Unknown Error. Re-try in 15 secs.")
        time.sleep(15.)
      elif status == 1:
        print ("User %s offline" % username)
        # print (info)
        time.sleep(self.refresh)
      elif status == 0:

        # Just one video per day
        day=datetime.datetime.now().strftime("%Y-%m-%d")

        if username in self.last_record_date and day == self.last_record_date[username]:
          print('Ignoring user %s b/c this user already made a recording today' % username)
        else:

          print ("%s online, recording..." % username)
          filename = username + " - " + datetime.datetime.now().strftime("%Y-%m-%d %Hh%Mm%Ss") + " - " + (info['stream']).get("channel").get("status") + ".mp4"
          filename = "".join(x for x in filename if x.isalnum() or x in [" ", "-", "_", "."])
          recorded_filename = os.path.join(self.recorded_path, filename)
          subprocess.call(["streamlink", "--twitch-oauth-token", self.oauth_token, "twitch.tv/" + username, self.quality, "-o", recorded_filename, "--hls-duration", self.DURATION_LIMIT])

          print ("Recording completed. Fixing video file")
          if os.path.exists(recorded_filename):
            try:
              subprocess.call([self.ffmpeg_path, '-err_detect', 'ignore_err', '-i', recorded_filename, '-c', 'copy', os.path.join(self.processed_path, filename)])
              os.remove(recorded_filename)
            except Exception as e:
              print (e)
          else:
            print ("File not found.")

          print ("All done. Upload to S3..")
          subprocess.call(["aws", "s3", "mv", os.path.join(self.processed_path, filename), self.S3_PATH])
          # Mark the day of the last recording
          self.last_record_date[username] = day
          ## mv will delete source

        time.sleep(self.refresh)

      sys.stdout.flush()

  def run(self):

    for path in [self.recorded_path, self.processed_path]:
      if not os.path.exists(path):
        os.makedirs(path)

    self.refresh = max(self.refresh, 15.)

    print("Checking for", "every", self.refresh, "seconds. Record with", self.quality, "quality.")  
    self.loopcheck()

