#!/usr/bin/env python
import numpy as np
import pickle
import os
from PIL import Image
import time
from tqdm import tqdm
import shutil
from random import randint
import argparse

import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

import dataloader
from utils import *
from network import *

from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

#Largely based on the code from https://github.com/jeffreyhuang1/two-stream-action-recognition

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='Training spatial stream model on resnet101')
parser.add_argument('--epochs', required=True, type=int, metavar='N', help='number of total epochs')
parser.add_argument('--data_root', required=True, type=str, metavar='data root', help='data root')
parser.add_argument('--batch_size', required=True, type=int, metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--lr', default=1e-3, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--train_desc', required=True, type=str, 
                    metavar='training data description numpy file', 
                    help='training data description numpy file')
parser.add_argument('--test_desc', required=True, type=str, 
                    metavar='training data description numpy file', 
                    help='training data description numpy file')
parser.add_argument('--class_qty', required=True, type=int, metavar='# of classes', help='number of classes')
parser.add_argument('--class_qty_new', default=None, type=int, metavar='new # of classes', help='new # of classes')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--patience', default=3, type=int, metavar='N', help='manual epoch number (useful on restarts)')

def main():
    global arg
    args = parser.parse_args()
    print(args)

    #Prepare DataLoader
    train_loader_obj = dataloader.Spatial_Dataloader(
                        is_val=False,
                        batch_size=args.batch_size,
                        num_workers=1,
                        root_dir=args.data_root,
                        desc_numpy_file=args.train_desc)

    val_loader_obj = dataloader.Spatial_Dataloader(
                        is_val=True,
                        batch_size=args.batch_size,
                        num_workers=1,
                        root_dir=args.data_root,
                        desc_numpy_file=args.test_desc)
    
   
    train_loader = train_loader_obj.get_loader()
    val_loader = val_loader_obj.get_loader()
  
    #Model 
    model = Spatial_CNN(
                        nb_epochs=args.epochs,
                        lr=args.lr,
                        patience=args.patience,
                        batch_size=args.batch_size,
                        resume=args.resume,
                        start_epoch=args.start_epoch,
                        evaluate=args.evaluate,
                        train_loader=train_loader,
                        test_loader=val_loader,
                        nb_classes=args.class_qty,
                        nb_classes_new=args.class_qty_new)
    #Training
    model.run()

class Spatial_CNN():
    def __init__(self, nb_epochs, lr, patience, batch_size, resume, start_epoch, evaluate, train_loader, test_loader, 
                 nb_classes, nb_classes_new):
        self.nb_epochs=nb_epochs
        self.lr=lr
        self.patience=patience
        self.batch_size=batch_size
        self.resume=resume
        self.start_epoch=start_epoch
        self.evaluate=evaluate
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.best_auc_score=0
        self.nb_classes=nb_classes
        self.nb_classes_new=nb_classes_new

    def build_model(self):
        print('==> Build model and setup loss and optimizer')
        #build model
        self.model = resnet101(pretrained=True, nb_classes=self.nb_classes, channel=3).cuda()
        #Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=self.patience,verbose=True)
    
    def resume_and_evaluate(self):
        if self.resume:
            if os.path.isfile(self.resume):
                print("==> loading checkpoint '{}'".format(self.resume))
                checkpoint = torch.load(self.resume)
                self.start_epoch = checkpoint['epoch']
                self.best_auc_score = checkpoint['best_auc_score']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("==> loaded checkpoint '{}' (epoch {}) (best_auc_score {})"
                  .format(self.resume, checkpoint['epoch'], self.best_auc_score))

                if self.nb_classes_new is not None:
                  print('Resetting the number of output classes to:', self.nb_classes_new)
                  self.model.reset_nb_classes(self.nb_classes_new)
                  if self.nb_classes_new == 2:
                    self.model.set_freeze(True)
                  self.model.cuda()
                  self.nb_classes = self.nb_classes_new
                  self.best_auc_score = 0

                  print('Reseting optimizer to have LR:', self.lr)
                  self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)
                  self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=self.patience,verbose=True)

            else:
                print("==> no checkpoint found at '{}'".format(self.resume))
        if self.evaluate:
            self.epoch = 0
            auc_score, prec1, val_loss = self.validate_1epoch()
            return

    def run(self):
        self.build_model()
        self.resume_and_evaluate()
        cudnn.benchmark = True
        
        for self.epoch in range(self.start_epoch, self.nb_epochs):
            self.train_1epoch()
            auc_score, prec1, val_loss = self.validate_1epoch()
            is_best = auc_score > self.best_auc_score
            #lr_scheduler
            self.scheduler.step(val_loss)
            # save model
            if is_best:
                self.best_auc_score = auc_score
                with open('record/spatial/spatial_video_preds.pickle','wb') as f:
                    pickle.dump(self.dic_video_level_preds,f)
                f.close()
            
            save_checkpoint({
                'epoch': self.epoch,
                'state_dict': self.model.state_dict(),
                'best_auc_score': self.best_auc_score,
                'optimizer' : self.optimizer.state_dict()
            },is_best,'record/spatial/checkpoint.pth.tar','record/spatial/model_best.pth.tar')

    def train_1epoch(self):
        print('==> Epoch:[{0}/{1}][training stage]'.format(self.epoch, self.nb_epochs))
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        #switch to train mode
        self.model.train()    
        end = time.time()
        # mini-batch training
        progress = tqdm(self.train_loader)
        for i, (data_dict,label) in enumerate(progress):

    
            # measure data loading time
            data_time.update(time.time() - end)
            
            label = label.cuda(async=True)
            target_var = Variable(label).cuda()

            # compute output
            for i in range(len(data_dict)):
                key = 'img'+str(i)
                data = data_dict[key]
                input_var = Variable(data).cuda()
                if i == 0:
                  output = self.model(input_var)
                else:
                  output += self.model(input_var)

            loss = self.criterion(output, target_var)

            # measure accuracy and record loss
            prec1, = accuracy(output.data, label, topk=(1, ))
            losses.update(loss.item(), data.size(0))
            top1.update(prec1.item(), data.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        
        info = {'Epoch':[self.epoch],
                'Batch Time':[round(batch_time.avg,3)],
                'Data Time':[round(data_time.avg,3)],
                'Loss':[round(losses.avg,5)],
                'AUC':[-1],
                'Prec@1':[round(top1.avg,4)],
                'lr': self.optimizer.param_groups[0]['lr']
                }
        record_info(info, 'record/spatial/rgb_train.csv','train')

    def validate_1epoch(self):
        print('==> Epoch:[{0}/{1}][validation stage]'.format(self.epoch, self.nb_epochs))
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        # switch to evaluate mode
        self.model.eval()
        self.dic_video_level_preds={}
        end = time.time()
        progress = tqdm(self.test_loader)
        self.video_to_label = {}
        for i, (video_names, data, video_labels) in enumerate(progress):
            
            video_labels = video_labels.cuda(async=True)
            data_var = Variable(data).cuda(async=True)
            label_var = Variable(video_labels).cuda(async=True)

            # compute output
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
                else:
                    self.dic_video_level_preds[video] += preds[j,:]
                    
        #Frame to video level accuracy
        video_top1, video_loss, f1_0, f1_1, auc_score = self.frame2_video_level_accuracy()
        info = {'Epoch':[self.epoch],
                'Batch Time':[round(batch_time.avg,3)],
                'Loss':[round(video_loss,5)],
                'AUC':[round(auc_score, 3)],
                'Prec@1':[round(video_top1,3)]
                }
        record_info(info, 'record/spatial/rgb_test.csv','test')
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

        loss = self.criterion(Variable(video_level_preds).cuda(), Variable(video_level_labels).cuda())    
        top1, = accuracy(video_level_preds, video_level_labels, topk=(1,))     
                            
        top1 = float(top1.numpy())

        if self.nb_classes == 2:
          f1_0 = f1_score(video_level_labels.numpy(), video_level_preds_class, pos_label=0)
          f1_1 = f1_score(video_level_labels.numpy(), video_level_preds_class, pos_label=1)
          auc_score = roc_auc_score(video_level_labels, video_level_preds.numpy()[:,1])
        else:
          f1_0 = f1_score(video_level_labels.numpy(), video_level_preds_class, average='macro')
          f1_1 = -1
          auc_score = roc_auc_score(label_binarize(video_level_labels, classes=np.arange(self.nb_classes)), 
                                    video_level_preds.numpy(), average='macro')
            
        return top1, loss.cpu().item(), f1_0, f1_1, auc_score






if __name__=='__main__':
    main()
