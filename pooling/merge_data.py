#!/usr/bin/env python3
"""
Pool all predictions based on the train set models
audio
video - rgb
video - flow
chat - text
chat - temporal

Compute statistics on validation set for each source
AUC, F1

Return model dictionary

"""

import os, sys
import string
import argparse
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, classification_report
import numpy as np
import csv
from tqdm import tqdm
import json

DEBUG = True

class DataAggregator:

  def __init__(self, min = 0.0001,max=0.9999):

    self.players = {}
    self.min = min
    self.max = max


  def read_labels(self,file, type):

    print("Reading ", file)

    with open(file, "r") as fl:

      reader = csv.reader(fl)
      next(reader)

      for i, line in tqdm(enumerate(reader)):
        if len(line) == 6:
          id, rank, _, _, _, _ = line
        else:
          raise ValueError(f"{file} line has {len(line)} values")

        if id not in self.players:
          self.players[id]={}

        self.players[id]["rank"] = int(rank)
        self.players[id]["set"] = type

  def read_audio(self,file):

    print("Reading ", file)


    with open(file,"r") as fl:

      reader = csv.reader(fl)

      for i, line in tqdm(enumerate(reader)):

        if len(line) == 2:
          id_raw, pred = line
        else:
          raise ValueError(f"{file} line has {len(line)} values")

        id = id_raw.split()[0]

        if id not in self.players:
          self.players[id]={}
          self.players[id]["set"] = "none"

        if "audio" not in self.players[id]:
          #print(id, self.players[id]["set"])
          self.players[id]["audio"]=[]

        self.players[id]["audio"].append(self.logit(float(pred)))


  def read_chat(self,file,type):

    print("Reading ", file)

    with open(file,"r") as fl:

      reader = csv.reader(fl)
      next(reader)

      for i, line in tqdm(enumerate(reader)):

        if len(line) == 2:
          id, pred = line
        else:
          raise ValueError(f"{file} line has {len(line)} values")

        if id not in self.players:
          print(id)
          self.players[id]={}
          self.players[id]["set"] = "none"

        if type not in self.players[id]:
          print(id, self.players[id]["set"])
          self.players[id][type] = []

        self.players[id][type].append(self.logit(float(pred)))


  def read_video(self,file,type):

    print("Reading ", file)

    #dropl = ['1perkyomg','nzer0name','msxshots','tyler_a_lot','xdustee','humbabeast']

    with open(file, "r") as fl:

      dict = json.load(fl)

      for elem in tqdm(dict):

        id = elem["twitch"]

        #if id in dropl:
        #  continue

        if id not in self.players:
          print(id)
          self.players[id] = {}
          self.players[id]["set"] = "none"

        if type not in self.players[id]:
          print(id, self.players[id]["set"])
          self.players[id][type] = []

        self.players[id][type].append(elem["logits"][1])

  def logit(self, p):
    stable_prob =  min(max(p,self.min),self.max)
    return np.log(stable_prob / (1-stable_prob))

  def get_data(self, set, feature, pool):

    preds = []
    labels = []

    for id in self.players:

      if self.players[id]["set"] == set:

        if feature not in self.players[id]:
          continue

        if pool:

          pred = []
          for p in self.players[id][feature]:
            pred.append(p)

          labels.append(self.players[id]["rank"])
          preds.append(np.mean(pred))

        else:

          for pred in self.players[id][feature]:
            labels.append(self.players[id]["rank"])
            preds.append(pred)

    return preds, labels

  def evaluate(self,set,feature, pool):

    scores = {}

    preds, labels = self.get_data(set,feature, pool)

    scores['N'] = len(preds)
    scores['AUC'] = roc_auc_score(labels, preds)

    if set=="valid":
      preds_tr, labels_tr = self.get_data("train", feature, pool)
    else:
      preds_tr, labels_tr = preds, labels

    #print(preds)
    #print(preds_tr)

    thresh = self.optim_thresh(preds_tr,labels_tr)
    scores['thresh'] = thresh
    scores['min'] = min(preds)
    scores['max'] = max(preds)
    scores['F1'] = f1_score(labels,preds >= thresh)
    scores['precision'] = precision_score(labels,preds >= thresh)
    scores['recall'] = recall_score(labels,preds >= thresh)

    #print(classification_report(labels,preds >= thresh))
    #print(labels)
    preds1 = [preds[i] for i in range(len(labels)) if labels[i]==1]
    preds0 = [preds[i] for i in range(len(labels)) if labels[i]==0]
    print("Average for label=1",np.mean(preds1))
    print("Average for label=0",np.mean(preds0))

    return scores

  def optim_thresh(self, preds, labels):

    best_score = -99999
    best_thres = -99999

    mn = np.quantile(preds,0.10)
    mx = np.quantile(preds,0.90)

    for x in np.linspace(mn, mx, num=999):

      if (sum(preds >= x) == 0) or (sum(preds <= x) == 0):
        print("skip....")
        continue

      scr = f1_score(labels, preds >= x)
      if scr > best_score:
        best_thres = x
        best_score = scr

    return best_thres

  def prep_data(self, pool):
    '''
    id
    P_obs
    P_new
    P
    r

    L_aud
    L_txt
    L_tmp
    L_img
    L_mov

    seg_aud
    seg_txt
    seg_tmp
    seg_img
    seg_mov

    logits_aud
    logits_txt
    logits_tmp
    logits_img
    logits_mov

    :param pool:
    :return: dictionary
    '''

    # Create the array of player ids in the right sequence
    id_obs = []
    id_new = []
    r = []
    seg_aud = []
    seg_txt = []
    seg_tmp = []
    seg_img = []
    seg_mov = []

    logits_aud = []
    logits_txt = []
    logits_tmp = []
    logits_img = []
    logits_mov = []

    # Loop through players to be modeled with ranks
    for key, pl in self.players.items():

      # check if player has at least one data element
      add_flag = 0
      if 'audio' in pl: add_flag += 1
      if 'txt' in pl: add_flag += 1
      if 'tmp' in pl: add_flag += 1
      if 'img' in pl: add_flag += 1
      if 'mov' in pl: add_flag += 1
      if add_flag == 0: continue

      if pl["set"] == "valid":
        id_obs.append(key)
        r.append(pl["rank"])
      if pl["set"] == "train":
        id_new.append(key)

    id = id_obs + id_new
    #print(id)

    # Loop through players of interest to collect data

    for i in id:
      pl = self.players[i]

      if "audio" in pl:
        seg_aud.append(len(pl["audio"]))
        logits_aud.extend(pl["audio"])
      else:
        seg_aud.append(0)

      if "txt" in pl:
        seg_txt.append(len(pl["txt"]))
        logits_txt.extend(pl["txt"])
      else:
        seg_txt.append(0)

      if "tmp" in pl:
        seg_tmp.append(len(pl["tmp"]))
        logits_tmp.extend(pl["tmp"])
      else:
        seg_tmp.append(0)

      if "img" in pl:
        seg_img.append(len(pl["img"]))
        logits_img.extend(pl["img"])
      else:
        seg_img.append(0)

      if "mov" in pl:
        seg_mov.append(len(pl["mov"]))
        logits_mov.extend(pl["mov"])
      else:
        seg_mov.append(0)

    #populate dictionary
    model_dict = {

      'id' : id,
      'P': len(id),
      'P_obs' : len(id_obs),
      'P_new' : len(id_new),
      'r' : r,

      'L_aud' : len(logits_aud),
      'L_txt': len(logits_txt),
      'L_tmp': len(logits_tmp),
      'L_img': len(logits_img),
      'L_mov': len(logits_mov),

      'seg_aud' : seg_aud,
      'seg_txt' : seg_txt,
      'seg_tmp' : seg_tmp,
      'seg_img': seg_img,
      'seg_mov': seg_mov,

      'logits_aud' : logits_aud,
      'logits_txt' : logits_txt,
      'logits_tmp' : logits_tmp,
      'logits_img': logits_img,
      'logits_mov': logits_mov
    }

    return model_dict

def main(argv):

  parser = argparse.ArgumentParser(description='Basic Processing and Evaluation')
  parser.add_argument('--labelpath', type=str,
                      required=False,
                      default = '../label_data/',
                      help = "Path to label data")
  parser.add_argument('--labeltrain', type=str,
                      required=False,
                      default = 'train_small.csv',
                      help = "Train labels")
  parser.add_argument('--labelvalid', type=str,
                      required=False,
                      default = 'valid_small.csv',
                      help = 'Valid labels')
  parser.add_argument('--datapath', type=str,
                      required=False,
                      default = 'raw_predictions/',
                      help = 'Data folder')
  parser.add_argument('--audio', type=str,
                      required = False,
                      default = 'proba_audio.txt',
                      help = 'Audio train')
  parser.add_argument('--chat_txt_trn', type=str,
                      required = False,
                      default = 'chat_train.csv',
                      help = 'Chat text train')
  parser.add_argument('--chat_txt_val', type=str,
                      required = False,
                      default = 'chat_val.csv',
                      help = 'Chat text valid')
  parser.add_argument('--chat_tmp_trn', type=str,
                      required = False,
                      default = 'chat_train_temporal.csv',
                      help = 'Chat temporal train')
  parser.add_argument('--chat_tmp_val', type=str,
                      required = False,
                      default = 'chat_val_temporal.csv',
                      help = 'Chat temoral valid')
  parser.add_argument('--img_trn', type=str,
                      required = False,
                      default = 'train_spatial_train.json',
                      help = 'Image train')
  parser.add_argument('--img_val', type=str,
                      required = False,
                      default = 'train_spatial_valid.json',
                      help = 'Image valid')
  parser.add_argument('--mov_trn', type=str,
                      required = False,
                      default = 'train_motion_train.json',
                      help = 'Motion train')
  parser.add_argument('--mov_val', type=str,
                      required = False,
                      default = 'train_motion_valid.json',
                      help = 'Motion valid')
  parser.add_argument('--output', type=str,
                      required = False,
                      default = '../notebooks/predictions/data.json',
                      help = 'Processed data for Bayesian model')



  args = parser.parse_args(argv)
  print(args)

  data = DataAggregator()
  data.read_labels(os.path.join(args.labelpath,args.labeltrain),"train")
  data.read_labels(os.path.join(args.labelpath,args.labelvalid),"valid")

  data.read_audio(os.path.join(args.datapath,args.audio))

  data.read_chat(os.path.join(args.datapath,args.chat_txt_trn),"txt")
  data.read_chat(os.path.join(args.datapath,args.chat_tmp_trn),"tmp")

  data.read_chat(os.path.join(args.datapath,args.chat_txt_val),"txt")
  data.read_chat(os.path.join(args.datapath,args.chat_tmp_val),"tmp")

  data.read_video(os.path.join(args.datapath, args.img_trn), "img")
  data.read_video(os.path.join(args.datapath, args.mov_trn), "mov")

  data.read_video(os.path.join(args.datapath, args.img_val), "img")
  data.read_video(os.path.join(args.datapath, args.mov_val), "mov")

  if DEBUG:
    for key, pl in data.players.items():
      if pl["set"] == "valid":
        print("=====",key,"========")
        print("rank", pl['rank'])
        print("audio", pl['audio'] if 'audio' in pl else None)
        print("txt", pl['txt'] if 'txt' in pl else None)
        print("tmp", pl['tmp'] if 'tmp' in pl else None)
        print("img", pl['img'] if 'img' in pl else None)
        print("mov", pl['mov'] if 'mov' in pl else None)


  summary = {}
  summary["valid"] = {}
  summary["train"] = {}
  summary["none"] = {}

  for key, elem in summary.items():
    elem["N"] = 0
    elem["audf"] = 0
    elem["txtf"] = 0
    elem["tmpf"] = 0
    elem["imgf"] = 0
    elem["movf"] = 0

    elem["N1"] = 0
    elem["audf1"] = 0
    elem["txtf1"] = 0
    elem["tmpf1"] = 0
    elem["imgf1"] = 0
    elem["movf1"] = 0

    elem["N0"] = 0
    elem["audf0"] = 0
    elem["txtf0"] = 0
    elem["tmpf0"] = 0
    elem["imgf0"] = 0
    elem["movf0"] = 0


  #print(summary)

  for key, pl in data.players.items():

    summary[pl['set']]['N'] += 1
    summary[pl['set']]['audf'] += 1 if 'audio' in pl else 0
    summary[pl['set']]['txtf'] += 1 if 'txt' in pl else 0
    summary[pl['set']]['tmpf'] += 1 if 'tmp' in pl else 0
    summary[pl['set']]['imgf'] += 1 if 'img' in pl else 0
    summary[pl['set']]['movf'] += 1 if 'mov' in pl else 0

    if 'rank' in pl:

      if pl['rank'] == 1:
        summary[pl['set']]['N1'] += 1
        summary[pl['set']]['audf1'] += 1 if 'audio' in pl else 0
        summary[pl['set']]['txtf1'] += 1 if 'txt' in pl else 0
        summary[pl['set']]['tmpf1'] += 1 if 'tmp' in pl else 0
        summary[pl['set']]['imgf1'] += 1 if 'img' in pl else 0
        summary[pl['set']]['movf1'] += 1 if 'mov' in pl else 0

      else:
        summary[pl['set']]['N0'] += 1
        summary[pl['set']]['audf0'] += 1 if 'audio' in pl else 0
        summary[pl['set']]['txtf0'] += 1 if 'txt' in pl else 0
        summary[pl['set']]['tmpf0'] += 1 if 'tmp' in pl else 0
        summary[pl['set']]['imgf0'] += 1 if 'img' in pl else 0
        summary[pl['set']]['movf0'] += 1 if 'mov' in pl else 0

  print("Valid", summary['valid'])
  print("Train", summary['train'])
  #print("None", summary['none'])

  print("Train audio pool", data.evaluate("train","audio",pool=True))
  print("Train txt pool", data.evaluate("train","txt",pool=True))
  print("Train tmp pool", data.evaluate("train","tmp",pool = True))
  print("Train img pool", data.evaluate("train","img", pool=True))
  print("Train mov pool", data.evaluate("train","mov", pool = True))

  print("Train audio raw", data.evaluate("train","audio",pool=False))
  print("Train txt raw", data.evaluate("train","txt",pool=False))
  print("Train tmp pool", data.evaluate("train","tmp",pool = True))
  print("Train img raw", data.evaluate("train","img", pool=False))
  print("Train mov raw", data.evaluate("train","mov", pool = False))

  print("Valid audio pool", data.evaluate("valid","audio",pool=True))
  print("Valid txt pool", data.evaluate("valid","txt",pool=True))
  print("Valid tmp pool", data.evaluate("valid","tmp",pool=True))
  print("Valid img pool", data.evaluate("valid","img",pool=True))
  print("Valid mov pool", data.evaluate("valid","mov",pool=True))

  print("Valid audio raw", data.evaluate("valid","audio",pool=False))
  print("Valid txt raw", data.evaluate("valid","txt",pool=False))
  print("Valid tmp pool", data.evaluate("valid","tmp",pool=True))
  print("Valid img raw", data.evaluate("valid","img",pool=False))
  print("Valid mov raw", data.evaluate("valid","mov",pool=False))

  # Return and write dictionary for the modeling
  mod_dict = data.prep_data(pool=False)

  with open(args.output, "w") as write_file:
    json.dump(mod_dict, write_file)

if __name__ == '__main__':
  main(sys.argv[1:])
