#!/usr/bin/env python
import numpy as np
import json
import pickle, sys, os
from PIL import Image
import time
import tqdm
import shutil
from random import randint
import argparse

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import *
from network import *
import dataloader

from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

from motion_cnn import SEQ_FRAME_QTY

#Largely based on the code from https://github.com/jeffreyhuang1/two-stream-action-recognition

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description='Training motion stream model based on resnet101')
parser.add_argument('--data_root', required=True, type=str, metavar='data root', help='data root')
parser.add_argument('--pred_file', required=True, type=str, metavar='pred. file', help='pred. file')
parser.add_argument('--test_desc', required=True, type=str, 
                    metavar='training data description numpy file', 
                    help='training data description numpy file')
parser.add_argument('--batch_size', default=32, type=int, metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--cnn_type', required=True, type=str, metavar='CNN type motion or spatial', help='CNN type motion or spatial')
parser.add_argument('--test_frame_qty', default=10, type=int, metavar='N', help='# of frames per video to generate predictions')
parser.add_argument('--class_qty', required=True, type=int, metavar='# of classes', help='number of classes')
parser.add_argument('--with_svm', dest='with_svm', action='store_true', help='if specified, use SVM instead of the last projection layer')

def main():
    global arg
    args = parser.parse_args()
    print(args)


    eval = Eval(args)
    eval.validate_1epoch()
    eval.save_preds()

class Eval:

    def __init__(self, args):
        self.nb_classes = args.class_qty
        if args.cnn_type == 'motion':
            test_loader_obj = dataloader.Motion_DataLoader(
                        is_val=True,
                        batch_size=args.batch_size,
                        num_workers=1,
                        root_dir=args.data_root,
                        desc_numpy_file=args.test_desc,
                        in_channel=SEQ_FRAME_QTY,
                        val_sample_qty=args.test_frame_qty)
            self.model = resnet101(pretrained=False, nb_classes=self.nb_classes, channel=2*SEQ_FRAME_QTY).cuda()
        elif args.cnn_type == 'spatial':
            test_loader_obj = dataloader.Spatial_Dataloader(
                        is_val=True,
                        batch_size=args.batch_size,
                        num_workers=1,
                        root_dir=args.data_root,
                        desc_numpy_file=args.test_desc,
                        val_sample_qty=args.test_frame_qty)
            self.model = resnet101(pretrained=False, nb_classes=self.nb_classes, channel=3).cuda()
        else:
            raise Exception('Wrong CNN type:', args.cnn_type)
        self.pred_file = args.pred_file
        self.with_svm = args.with_svm
   
        self.test_loader = test_loader_obj.get_loader()

        model_root_subdir = 'record' if not self.with_svm else 'record_triplet'
  
        fn = f'{model_root_subdir}/{args.cnn_type}/model_best.pth.tar'
        print('Loading the main model from file:', fn)
        checkpoint = torch.load(fn)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.criterion = nn.CrossEntropyLoss().cuda()

        if self.with_svm:
          fn = f'{model_root_subdir}/{args.cnn_type}/clf.bin'
          print('Loading the additional model from file:', fn)
          with open(fn, 'rb') as f:
            self.clf = pickle.load(f)


    def save_preds(self):
        res = []

        for video in self.dic_video_level_preds.keys():
            label = self.video_to_label[video].item()
            qty = float(self.dic_video_level_preds_qty[video])
            logits = self.dic_video_level_preds[video] / qty
            twitch = os.path.basename(video).strip().split()[0] 
            #print(twitch, 'label:', label, qty, video)
            res.append( {'twitch' : twitch, 'video' : video, 'logits' : numpy_to_list(logits), 'label' : label } )

        with open(self.pred_file, 'w') as f:
            json.dump(res, f)
        

    def validate_1epoch(self):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        # switch to evaluate mode
        self.model.eval()
        self.dic_video_level_preds={}
        self.dic_video_level_preds_qty={}
        end = time.time()
        progress = tqdm(self.test_loader)
        self.video_to_label = {}
        for i, (video_names, data, video_labels) in enumerate(progress):
            
            video_labels = video_labels.cuda(async=True)
            data_var = Variable(data).cuda(async=True)
            label_var = Variable(video_labels).cuda(async=True)

            # compute output
            if self.with_svm:
              output = torch.Tensor(self.clf.predict_proba(self.model.forward_no_last(data_var).detach().cpu().numpy()))
            else:
              output = self.model(data_var)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            #Calculate video level prediction
            preds = output.data.cpu().numpy()
            nb_data = preds.shape[0]
            for j in range(nb_data):
                video = video_names[j]
                self.video_to_label[video] = video_labels[j]
                if video not in self.dic_video_level_preds.keys():
                    self.dic_video_level_preds[video] = preds[j,:]
                    self.dic_video_level_preds_qty[video] = 1 
                else:
                    self.dic_video_level_preds[video] += preds[j,:]
                    self.dic_video_level_preds_qty[video] += 1
                    
        #Frame to video level accuracy
        video_top1, video_loss, f1_0, f1_1, auc_score = self.frame2_video_level_accuracy()
        print('AUC %g' % auc_score)
        print('F-scores %g %g' % (f1_0, f1_1))
        print('')
        return auc_score, video_top1, video_loss

    def frame2_video_level_accuracy(self):
            
        correct = 0
        video_level_preds = np.zeros((len(self.dic_video_level_preds), self.nb_classes))
        video_level_preds_class = np.zeros(len(self.dic_video_level_preds), dtype=int)
        video_level_labels = np.zeros(len(self.dic_video_level_preds))
        ii=0
        for name in sorted(self.dic_video_level_preds.keys()):

            preds = self.dic_video_level_preds[name]
            label = int(self.video_to_label[name])
                
            video_level_preds[ii,:] = preds
            video_level_labels[ii] = label
            video_level_preds_class[ii] = np.argmax(preds)
            ii+=1         
            if np.argmax(preds) == (label):
                correct+=1

        #top1 
        video_level_labels = torch.from_numpy(video_level_labels).long()
        video_level_preds = torch.from_numpy(video_level_preds).float()

        #loss = self.criterion(Variable(video_level_preds).cuda(), Variable(video_level_labels).cuda())    
        loss = torch.FloatTensor([0])
        top1, = accuracy(video_level_preds, video_level_labels, topk=(1,))     
                            
        top1 = float(top1.numpy())

        f1_0 = f1_1 = auc_score = 0

        # Can't really run eval b/c e.g. on train we have 10 classes, but predict only 2 

        #if self.nb_classes == 2:
          #f1_0 = f1_score(video_level_labels.numpy(), video_level_preds_class, pos_label=0)
          #f1_1 = f1_score(video_level_labels.numpy(), video_level_preds_class, pos_label=1)
          #auc_score = roc_auc_score(video_level_labels, video_level_preds.numpy()[:,1])
        #else:
          #f1_0 = f1_score(video_level_labels.numpy(), video_level_preds_class, average='macro')
          #f1_1 = -1
          #auc_score = roc_auc_score(label_binarize(video_level_labels, classes=np.arange(self.nb_classes)), 
                                    #video_level_preds.numpy(), average='macro')
            
        return top1, loss.cpu().item(), f1_0, f1_1, auc_score

if __name__=='__main__':
    main()
