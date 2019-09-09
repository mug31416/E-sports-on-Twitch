#!/usr/bin/env python3
import os, pdb, sys, time, datetime
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18
from pytorch_data import *
from finetune_resnet import *
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler, RandomSampler
from modeling_common import *
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

USE_WEIGHTS = True

REPORT_SAVE_INTERVAL=5

def run_inference_for_epoch(model, optim, data_loaders, loss):
  """
  runs the inference algorithm for an epoch
  """

  epoch_loss = 0.

  for xs, ys in tqdm(data_loaders["train"]):

    optim.zero_grad()

    model.train(True)

    out = model(xs)

    # run the inference
    batch_avg_loss = loss(out, ys)
    #print('batch_loss', batch_avg_loss.data)
    batch_avg_loss.backward()

    optim.step()

    epoch_loss += batch_avg_loss.data.cpu()

    model.train(False)

  return epoch_loss.data.numpy()


def image_collate(batch):
  x_seq, y_seq = zip(*[(d, l) for (d, l) in batch])
  return torch.stack(x_seq),torch.cat(y_seq)

def image_collate_cuda(batch):
  x_seq, y_seq = zip(*[(d, l) for (d, l) in batch])
  return torch.stack(x_seq).cuda(), torch.cat(y_seq).cuda()

def setup_data_loaders(label_file_dict, file_dir, keyframe_interval, subset_id, use_cuda, cache_file_dir, batch_size, **kwargs):

  if 'num_workers' not in kwargs:
    kwargs = {'num_workers': 0, 'pin_memory': False}

  cached_data = {}
  loaders = {}

  for mode, label_file in label_file_dict.items():

    cached_data[mode] = VideoFramesTrain(label_file=label_file,
                                file_dir=file_dir,
                                keyframe_interval=keyframe_interval,
                                subset_id=subset_id,
                                cache_file_prefix=os.path.join(cache_file_dir, mode + '_cache'))

    collate_func = image_collate_cuda if use_cuda else image_collate

    if mode in ["test","valid"] :

      print(mode)
      print('# of data points', len(cached_data[mode]))
      loaders[mode] = DataLoader(cached_data[mode],
                                 batch_size=batch_size, shuffle=False,
                                 collate_fn = collate_func,
                                 **kwargs)

    else:

      print(mode)
      print('# of data points', len(cached_data[mode]))

      if USE_WEIGHTS:
        wghts = cached_data[mode].get_weights()
        print('Using weights', wghts, ' original weights')
        sampler = WeightedRandomSampler(wghts, len(cached_data[mode]), replacement=True)
      else:
        print('Using merely uniform sampling')
        sampler = RandomSampler(cached_data[mode], replacement=False)

      loaders[mode] = DataLoader(cached_data[mode],
                                   batch_size=batch_size,
                                   shuffle=True,
                                   #sampler=sampler, # Runs out of memory
                                    collate_fn=collate_func,
                                   **kwargs)

  return loaders, cached_data


def set_seed_all(seed, cuda):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if cuda:
    torch.cuda.manual_seed_all(seed)

def get_predictions(data_loader, model, invClassDict):

  predictions = []

  model.train(False)

  for (xs, _) in data_loader:

    print('xs size', xs.size())

    out = torch.argmax(model(xs), dim=1, keepdim=False) \
          + torch.ones(xs.size(0), device=xs.device, dtype=torch.long)

    outNumpy = out.cpu().data.numpy()

    for idx in outNumpy:
      predictions.append(invClassDict[idx])

  return np.array(predictions)


def get_scores(data_loader, model):
  """
  compute the error
  """
  pred = []
  target = []
  pred_no_zero = []


  model.train(False)

  for (xs, ys) in data_loader:

    mout = model(xs)
    if len(mout.size()) == 1:
      mout = mout.unsqueeze(0)
    out = torch.argmax(mout, dim=1, keepdim=False)
    out_no_zero = 1 + torch.argmax(mout[:,1:], dim=1, keepdim=False)

    pred.append(out.cpu().data.numpy())
    pred_no_zero.append(out_no_zero.cpu().data.numpy())
    target.append(ys.cpu().data.numpy())

  pred = np.hstack(pred)
  pred_no_zero = np.hstack(pred_no_zero)
  target = np.hstack(target)
  idx_no_zero = target > 0

  pos_ratio = np.sum(target == 1)/len(target)

  # Trying random baseline
  if False:
    p = 1 - 0.30973451327433627
    pred = (np.random.random(len(pred)) > p).astype(int)
    #pred = np.zeros(len(pred))

  return f1_score(target, pred, average='macro'), accuracy_score(target, pred), roc_auc_score(target, pred), pos_ratio

def print_and_log(file_handle, text):
  print(text + '\n')
  file_handle.write(text + '\n')


def main(args):

  set_seed_all(args.seed, args.use_cuda)

  shuffled_subset_ids = np.arange(args.subset_qty)

  set_seed_all(int(time.time()), args.use_cuda)

  np.random.shuffle(shuffled_subset_ids)

  for subset_id in [15]:#shuffled_subset_ids:

    print('Current subset:', subset_id)

    label_file_dict = {
      'train': args.train_list,
      'valid': args.valid_list
    }

    data_loaders, cached_data = setup_data_loaders(label_file_dict,
                                    args.file_dir, 
                                    args.keyframe_interval, subset_id,
                                    args.use_cuda,
                                    args.cache_file_dir,
                                    pin_memory = False,
                                    batch_size = args.bs)

    train_data = cached_data['train']
    num_train = len(train_data)

    if args.seed is not None:
      set_seed_all(args.seed, args.use_cuda)

    freeze_old = not args.unfreeze_resnet
    if os.path.exists(os.path.join(os.getcwd(),args.outmod,'model.pt')):
      print('Loading model %s' % 'model.pt')
      model = torch.load(os.path.join(os.getcwd(),args.outmod,'model.pt'))
      model.reset_dropout(args.drop_out)
      model.change_freeze(freeze_old)

    else:

      model = ResNetExtractor(feat_dim=args.feat_dim, drop_out=args.drop_out, out_dim=CLASS_QTY, freeze_old=freeze_old)

    print(model)



    #optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)

    w = torch.FloatTensor(train_data.get_class_weights())

    if args.use_cuda:
      model.cuda()
      w = w.cuda()

    loss = nn.CrossEntropyLoss(weight=w)
    #loss = nn.CrossEntropyLoss()
    try:

      logger = open(args.logfile, "w") if args.logfile else None

      # run inference for a certain number of epochs
      for i in range(0, args.n):

        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

        epoch_losses = \
            run_inference_for_epoch(model, optim, data_loaders, loss)

        avg_epoch_losses = epoch_losses / num_train

        str_loss = str(avg_epoch_losses)


        if i % REPORT_SAVE_INTERVAL == 0 or i + 1 == args.n:
          f1_train, acc_train, auc_train, pos_ratio_train = get_scores(data_loaders['train'], model)
          f1_valid, acc_valid, auc_valid, pos_ratio_valid = get_scores(data_loaders['valid'], model)

          str_print = f'{i} epoch: avg loss: {str_loss} f1 train: {f1_train} f1 valid: {f1_valid} acc train: {acc_train} acc valid.: {acc_valid} auc train: {auc_train} auc valid: {auc_valid} pos ratio train: {pos_ratio_train} pos ratio valid: {pos_ratio_valid}'
          print_and_log(logger, str_print)

          torch.save(model, os.path.join(os.getcwd(), args.outmod, 'model_' + st + '_epoc-' + str(i) + '.pt'))
        else:

          str_print = f'{i} epoch: avg loss: {str_loss}'
          print_and_log(logger, str_print)


      if args.n <= 0:

        f1_train, acc_train, auc_train, pos_ratio_train = get_scores(data_loaders['train'], model)
        f1_valid, acc_valid, auc_valid, pos_ratio_valid = get_scores(data_loaders['valid'], model)

        str_print = f'f1 valid: {f1_valid} acc train: {acc_train} acc valid.: {acc_valid} pos ratio train: {pos_ratio_train} auc train: {auc_train} auc valid: {auc_valid} pos ratio valid: {pos_ratio_valid}'

        print_and_log(logger, str_print)

      else:

        torch.save(model, os.path.join(os.getcwd(), args.outmod, 'model.pt'))



    finally:
      if args.logfile:
        logger.close()



try:
  if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train RESNET")

    parser.add_argument('-n',  default=1, type=int,
                        help="number of epochs to run")
    parser.add_argument('--lr',  default=0.01, type=float,
                        help="learning rate for the optimizer")
    parser.add_argument('--bs',  default=64, type=int,
                        help="number of examples to be considered in a batch")
    parser.add_argument('--drop_out',  default=0.25, type=float,
                        help="dropout")
    parser.add_argument('--logfile',  default="tmp.log", type=str,
                        help="filename for logging the outputs")
    parser.add_argument('--outmod', type=str,
                        default='models/hw2/',
                        help='Directory for models')
    parser.add_argument('--outpred', type=str,
                        default='predictions/hw2/', help='Directory for predictions')

    parser.add_argument('--train_list', type=str, required=True,
                        help='A list of train items and their labels')

    parser.add_argument('--valid_list', type=str,
                        required=True,
                        help='A list of validation items and their labels')

    parser.add_argument('--keyframe_interval', type=int,
                        default=20,
                        help='A number of frames to skip')

    parser.add_argument('--subset_qty', type=int,
                        required=True,
                        help='# of subsets')

    parser.add_argument('--feat_dim', type=int,
                        default=128,
                        help='A number of parameters in the last layer')

    parser.add_argument('--file_dir', type=str,
                        required=True,
                        help='Directory with video files')

    parser.add_argument('--cache_file_dir', type=str,
                        default=None,
                        help='Cache directory')

    parser.add_argument('--unfreeze_resnet',
                        action='store_true',
                        help='Do not freeze original ResNet layers')

    parser.add_argument('--seed',
                        default=0, type=int,
                        help='Random seed')

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
