#!/usr/bin/env python3
import os, pdb, sys, time, datetime
import numpy as np
import torch
from pytorch_data import *
import argparse
from sklearn.metrics import f1_score, accuracy_score

AVG_LOGITS=False

prev_dim = None

PRED_ONLY = True

def main(args):
  cuda = args.use_cuda
  model = torch.load(args.model_file)
  lab_ids = torch.LongTensor(np.arange(len(LABEL_DICT)))
  
  if cuda:
    model.cuda()
    lab_ids = lab_ids.cuda()
  print(model)


  save_res = []

  y_true = []
  y_pred = []

  label_file = args.item_list
  for line_num, file_name_prefix, label_str in get_labels(label_file):

    file_name = VideoFramesTest.cache_file_name(args.frame_file_dir, file_name_prefix)

    prob = None
    out = None

    if os.path.exists(file_name):

      lab = LABEL_DICT[label_str]
      print(file_name_prefix, '->', lab)
      y_true.append(lab)

      x = torch.FloatTensor(np.load(file_name))
      if cuda:
        x = x.cuda()
      y = model(x)
      y_feat = model.get_feat(x)

      # One-row result will miss this dummy dimension
      if len(y.shape) == 1:
        y = y.unsqueeze(0)
        y_feat = y_feat.unsqueeze(0)

      y_feat_mean = torch.mean(y_feat, dim=0)
      if AVG_LOGITS:
        mlog = torch.mean(y, dim=0)
        prob = torch.nn.functional.softmax(mlog, dim=0)
      else:
        sy = torch.nn.functional.softmax(y, dim=1)
        prob = torch.mean(sy, dim=0)
  

      if PRED_ONLY:
        out = (torch.argmax(prob) == lab_ids).cpu().detach().numpy().astype(int)
      else:
        out = torch.cat( (y_feat_mean, prob) )
        out = out.cpu().detach().numpy()
      

      prev_dim = out.shape[0]

      prob = prob.cpu().detach().numpy()

    else:

      print('Missing file:', file_name)
      continue

      if False:
        assert(prev_dim is not None) # Very unlikely any faulty file goes before
        # the first valid one. If unlucky set dim manually
        out = np.zeros(prev_dim)
        prob = np.zeros(len(LABEL_DICT))

    assert(out is not None)
    assert(prob is not None)

    if args.pred_dir is not None:
      out_file = os.path.join(args.pred_dir, file_name_prefix + '.npy')
      np.save(out_file, out)

    pred = prob.argmax()

    save_res.append( (file_name_prefix, pred) )

    #print(y)
    #print(sy)
    #print(torch.sum(sy, dim=1))
    #print(torch.sum(sy, dim=0))

    print(pred, lab)

    y_pred.append(pred)

  y_true = np.array(y_true)
  y_pred = np.array(y_pred)

  print('F-score macro: %g' % f1_score(y_true, y_pred, average='macro'))
  print('F-score micro: %g' % f1_score(y_true, y_pred, average='micro'))
  print('Accuracy: %g' % accuracy_score(y_true, y_pred))

  if args.pred_file is not None:
    with open(args.pred_file, 'w') as f:
      f.write('VideoID,Label\n')
      for vid, pred in save_res:
        f.write(vid + ',' + str(pred) + '\n')

  for lab in range(len(LABEL_DICT)):
    print('Label', lab)
    y_true_l = (y_true == lab).astype(int)
    y_pred_l = (y_pred == lab).astype(int)

    print('F-score: %g' % f1_score(y_true_l, y_pred_l, average='binary'))

try:
  if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate predictions based on extracted frames")

    parser.add_argument('--item_list', type=str, required=True,
                        help='A list of train/test/val items and their labels')

    parser.add_argument('--model_file', type=str,
                        required=True,
                        help='Model file')

    parser.add_argument('--pred_file', type=str,
                        default=None,
                        help='Prediction file')

    parser.add_argument('--pred_dir', type=str,
                        default=None,
                        help='Prediction directory')

    parser.add_argument('--frame_file_dir', type=str,
                        default=None,
                        help='Frame file directory')

    parser.add_argument('--use_cuda',
                        action='store_true',
                        help='Use cuda?')

    args = parser.parse_args()

    main(args)

except:
  # tb is traceback
  exType, value, tb = sys.exc_info()
  print(value)
  print(tb)
  pdb.post_mortem(tb)
