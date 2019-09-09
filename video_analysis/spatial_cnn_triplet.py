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

from dataloader import Spatial_Dataloader
from utils import *
from network import *
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

from dataloader.triplet_loss import OnlineTripletLoss, RandomNegativeTripletSelector

DUMMY_CLASS_QTY=2

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
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--patience', default=3, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--reset_auc', help='reset best AUC', action='store_true')

def main():
    global arg
    args = parser.parse_args()
    print(args)

    #Prepare DataLoader
    train_loader_obj = Spatial_Dataloader(
                        is_val=False,
                        batch_size=args.batch_size,
                        num_workers=1,
                        root_dir=args.data_root,
                        desc_numpy_file=args.train_desc,
                        for_triplet_loss=True)

    val_loader_obj = Spatial_Dataloader(
                        is_val=True,
                        batch_size=args.batch_size,
                        num_workers=1,
                        root_dir=args.data_root,
                        desc_numpy_file=args.test_desc,
                        for_triplet_loss=True)
    
   
    train_loader = train_loader_obj.get_loader()
    val_loader = val_loader_obj.get_loader()
  
    #Model 
    model = Spatial_CNN(nb_epochs=args.epochs,
                        lr=args.lr,
                        patience=args.patience,
                        batch_size=args.batch_size,
                        resume=args.resume,
                        start_epoch=args.start_epoch,
                        evaluate=args.evaluate,
                        train_loader=train_loader,
                        test_loader=val_loader,
                        nb_classes=args.class_qty,
                        reset_auc=args.reset_auc)
    #Training
    model.run()

class Spatial_CNN():
    def __init__(self, nb_epochs, lr, patience, batch_size, resume, start_epoch, evaluate, train_loader, test_loader, 
                 nb_classes, reset_auc):
        self.nb_epochs=nb_epochs
        self.lr=lr
        self.patience=patience
        self.batch_size=batch_size
        self.resume=resume
        self.start_epoch=start_epoch
        self.evaluate=evaluate
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.nb_classes=nb_classes
        self.best_auc=0
        self.reset_auc = reset_auc

    def build_model(self):
        print('==> Build model and setup loss and optimizer')
        #build model
        self.model = resnet101(pretrained=True, nb_classes=DUMMY_CLASS_QTY, channel=3).cuda()
        #Loss function and optimizer
        margin=1
        self.criterion = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin)).cuda()

        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=self.patience,verbose=True)
    
    def resume_and_evaluate(self):
        if self.resume:
            if os.path.isfile(self.resume):
                print("==> loading checkpoint '{}'".format(self.resume))
                checkpoint = torch.load(self.resume)
                if 'best_auc' in checkpoint:
                  self.best_auc = checkpoint['best_auc']
                  if self.reset_auc:
                    print('Reseting best AUC to zero')
                    self.best_auc = 0
                    self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)
                    #self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
                    self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=self.patience,verbose=True)
                else:
                  self.best_auc = 0
                self.start_epoch = checkpoint['epoch']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("==> loaded checkpoint '{}' (epoch {}) (best_auc {})"
                  .format(self.resume, checkpoint['epoch'], self.best_auc))

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
            is_best = auc_score > self.best_auc
            #lr_scheduler
            self.scheduler.step(val_loss)
            # save model
            if is_best:
                self.best_auc = auc_score
                with open('record_triplet/spatial/spatial_video_preds.pickle','wb') as f:
                    pickle.dump(self.dic_video_level_preds, f)

                print('Saving the SVM model')
                with open('record_triplet/spatial/clf.bin', 'wb') as f:
                    pickle.dump(self.clf, f)
            
            save_checkpoint({
                'epoch': self.epoch,
                'state_dict': self.model.state_dict(),
                'best_auc': self.best_auc,
                'optimizer' : self.optimizer.state_dict()
            },is_best,'record_triplet/spatial/checkpoint.pth.tar','record_triplet/spatial/model_best.pth.tar')

    def train_1epoch(self):
        print('==> Epoch:[{0}/{1}][training stage]'.format(self.epoch, self.nb_epochs))
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        #switch to train mode
        self.model.train()    
        end = time.time()
        # mini-batch training
        progress = tqdm(self.train_loader)

        labels_all = []
        output_lst = []

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
                  output = self.model.forward_no_last(input_var)
                else:
                  output += self.model.forward_no_last(input_var)

            output /= float(len(data_dict))
            loss, triplet_qty = self.criterion(output, target_var)


            # record loss
            losses.update(loss.item() * triplet_qty, triplet_qty)

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            labels_all.extend(label.detach().cpu().numpy())
            output_lst.append(output.detach().cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        labels_all = np.array(labels_all)
        outputs_all = np.vstack(output_lst)
        assert(labels_all.shape[0] == outputs_all.shape[0])

        if True:
          self.clf = SVC(probability=True, gamma='scale')
          #self.clf = SGDClassifier(loss='log', n_jobs=8)
          #self.clf = SVC(probability=True, kernel='linear')
        else:
          n_estimators=8
          self.clf = OneVsRestClassifier(BaggingClassifier(
            SVC(kernel='linear', probability=True, class_weight='balanced'),
            max_samples=1.0 / n_estimators, n_estimators=n_estimators, n_jobs=n_estimators),
            n_jobs=n_estimators)
        self.clf.fit(outputs_all, labels_all)

        train_pred = self.clf.predict(outputs_all)
        train_acc = accuracy_score(train_pred, labels_all)

        info = {'Epoch':[self.epoch],
                'Batch Time':[round(batch_time.avg,3)],
                'Data Time':[round(data_time.avg,3)],
                'Loss':[round(losses.avg,5)],
                'AUC':[-1],
                'Prec@1':[round(train_acc, 3)],
                'lr': self.optimizer.param_groups[0]['lr']
                }

        #print('Training accuracy: %g' % train_acc)
        record_info(info, 'record_triplet/spatial/rgb_train.csv','train')

    def validate_1epoch(self):
        print('==> Epoch:[{0}/{1}][validation stage]'.format(self.epoch, self.nb_epochs))
        batch_time = AverageMeter()
        losses = AverageMeter()

        # switch to evaluate mode
        self.model.eval()
        self.dic_video_level_preds={}
        self.dic_video_level_preds_qty = {}
        end = time.time()
        progress = tqdm(self.test_loader)
        self.video_to_label = {}
        self.embed_dim = None
        for i, (video_names, data, video_labels) in enumerate(progress):
            
            video_labels = video_labels.cuda(async=True)
            data_var = Variable(data).cuda(async=True)
            #target_var = Variable(video_labels).cuda()

            # compute output
            output = self.model.forward_no_last(data_var)
            self.embed_dim = output.size(-1)
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

        for video in self.dic_video_level_preds:
          self.dic_video_level_preds[video] /= float(self.dic_video_level_preds_qty[video])

        assert(self.embed_dim is not None)

        #Frame to video level accuracy
        video_top1, video_loss, f1_0, f1_1, auc_score = self.frame2_video_level_accuracy()
        info = {'Epoch':[self.epoch],
                'Batch Time':[round(batch_time.avg,3)],
                'Loss':[round(video_loss,5)],
                'AUC':[round(auc_score, 3)],
                'Prec@1':[round(video_top1,3)]
                }
        record_info(info, 'record_triplet/spatial/rgb_test.csv','test')
        print('AUC %g' % auc_score)
        print('F-scores %g %g' % (f1_0, f1_1))
        print('')
        return auc_score, video_top1, video_loss


    def frame2_video_level_accuracy(self):
            
        correct = 0
        video_level_preds = np.zeros((len(self.dic_video_level_preds), self.embed_dim))
        video_level_labels = np.zeros(len(self.dic_video_level_preds))
        ii=0
        for name in sorted(self.dic_video_level_preds.keys()):

            preds = self.dic_video_level_preds[name]
            label = int(self.video_to_label[name])
                
            video_level_preds[ii,:] = preds
            video_level_labels[ii] = label
            ii+=1

            #video_level_preds_class[ii] = np.argmax(preds)
            #if np.argmax(preds) == (label):
            #  correct+=1

        #top1 
        video_level_labels = torch.from_numpy(video_level_labels).long()
        video_level_preds = torch.from_numpy(video_level_preds).float()

        loss, _ = self.criterion(Variable(video_level_preds).cuda(), Variable(video_level_labels).cuda())

        video_level_preds_class = self.clf.predict(video_level_preds)
        video_level_probs_class = self.clf.predict_proba(video_level_preds)

        top1 = accuracy_score(video_level_preds_class, video_level_labels.numpy())

        if self.nb_classes == 2:
          f1_0 = f1_score(video_level_labels.numpy(), video_level_preds_class, pos_label=0)
          f1_1 = f1_score(video_level_labels.numpy(), video_level_preds_class, pos_label=1)
          auc_score = roc_auc_score(video_level_labels.numpy(), video_level_probs_class[:,1])
        else:
          f1_0 = f1_score(video_level_labels.numpy(), video_level_preds_class, average='macro')
          f1_1 = -1
          auc_score = roc_auc_score(label_binarize(video_level_labels.numpy(),
                                    classes=np.arange(self.nb_classes)),
                                    video_level_probs_class, average='macro')

        return top1, loss.cpu().item(), f1_0, f1_1, auc_score





if __name__=='__main__':
    main()
