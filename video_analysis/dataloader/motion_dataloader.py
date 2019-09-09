import numpy as np
import pickle, os, sys
from PIL import Image
import time
import shutil
import random
import argparse

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .triplet_loss import BalancedBatchSampler

#Largely based on the code from https://github.com/jeffreyhuang1/two-stream-action-recognition

class motion_dataset(Dataset):  
    # Totally dislike this organization, but to keep compatible with the previous code.
    # dic.values() are always labels
    # dic.keys() are tuples: (<video-prefix-name>, <number>)
    # 1. In the training regime, the number is the total number of frames,
    #    each call to getitem selects several frames starting a random point in time.
    # 2. In the validation regime, the number is the ID of the frame to return.
    #    Thus, getitem would return always the same set of adjacent frames (unlike training)
    def __init__(self, dic, in_channel, root_dir, mode, transform=None):
        self.keys=list(dic.keys())
        self.values=list(dic.values())
        self.root_dir = root_dir
        self.transform = transform
        self.mode=mode
        self.in_channel = in_channel
        self.img_rows=224
        self.img_cols=224

    def stackopf(self, video):
        name = video
        u = os.path.join(self.root_dir, name % 'x')
        v = os.path.join(self.root_dir, name % 'y')
        
        flow = torch.FloatTensor(2*self.in_channel,self.img_rows,self.img_cols)
        i = int(self.clips_idx)


        for j in range(self.in_channel):
            idx = i + j
            idx = str(idx)
            frame_idx = 'frame'+ idx.zfill(6)
            h_image = u +'/' + frame_idx +'.jpg'
            v_image = v +'/' + frame_idx +'.jpg'
            
            imgH=(Image.open(h_image))
            imgV=(Image.open(v_image))

            H = self.transform(imgH)
            V = self.transform(imgV)

            
            flow[2*(j-1),:,:] = H
            flow[2*(j-1)+1,:,:] = V      
            imgH.close()
            imgV.close()  
        return flow

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        #print ('mode:',self.mode,'calling Dataset:__getitem__ @ idx=%d'%idx)
        
        if self.mode == 'train':
            video, nb_clips = self.keys[idx]
            self.clips_idx = random.randint(1, nb_clips)
            #self.clips_idx = min(100, nb_clips)
        elif self.mode == 'val':
            video,self.clips_idx = self.keys[idx]
        else:
            raise ValueError('There are only train and val mode')

        #print('####', video, self.clips_idx)

        label = int(self.values[idx])
        data = self.stackopf(video)

        return (video, data, label)

    def get_labels(self):
        return np.array([int(v) for v in self.values])


class Motion_DataLoader:
    def __init__(self, is_val, batch_size, num_workers, in_channel, root_dir, desc_numpy_file,
                 val_sample_qty=3, for_triplet_loss=False):

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.in_channel = in_channel
        self.data_path = root_dir
        self.for_triplet_loss=for_triplet_loss
        # frame_count is a dictionary mapping video file (pattern) to the number of video frames
        self.frame_count={}
        # all_videos is a dictionary mapping video file (pattern) to the label
        self.all_videos = {}
        for label, file_name_pat, frame_count in np.load(desc_numpy_file):
            self.frame_count[file_name_pat] = int(frame_count) 
            self.all_videos[file_name_pat] = int(label) 
        
        if is_val:
          self.loader = self.val(val_sample_qty)
        else:
          self.loader = self.train()
        
    def get_loader(self):
        return self.loader    

    def build_val_dict(self, val_sample_qty):

        self.video_dict = {}
        
        for video in self.all_videos:
            sampling_interval = int((self.frame_count[video]-self.in_channel+1)/val_sample_qty)
            for index in range(val_sample_qty):
                clip_idx = index*sampling_interval
                key = (video, clip_idx+1)
                self.video_dict[key] = self.all_videos[video]


    def build_training_dict(self):

        self.video_dict={}
        for video in self.all_videos:
            
            nb_clips = self.frame_count[video]-self.in_channel+1
            key = (video, nb_clips)
            self.video_dict[key] = self.all_videos[video] 
                            
    def train(self):
        self.build_training_dict()

        training_set = motion_dataset(dic=self.video_dict, in_channel=self.in_channel, root_dir=self.data_path,
            mode='train',
            transform = transforms.Compose([
            transforms.Scale([224,224]),
            transforms.ToTensor(),
            ]))
        print('==> Training data # frames:',len(training_set))

        if self.for_triplet_loss:

            train_labels = training_set.get_labels()
            class_qty = len(set(train_labels))

            train_loader = DataLoader(
                dataset=training_set,
                batch_sampler=BalancedBatchSampler(train_labels, n_classes=class_qty, n_samples=self.batch_size),
                num_workers=self.num_workers,
                pin_memory=True
                )

        else:

            train_loader = DataLoader(
                dataset=training_set,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True
                )

        return train_loader

    def val(self, val_sample_qty):
        self.build_val_dict(val_sample_qty)

        validation_set = motion_dataset(dic= self.video_dict, in_channel=self.in_channel, root_dir=self.data_path ,
            mode ='val',
            transform = transforms.Compose([
            transforms.Scale([224,224]),
            transforms.ToTensor(),
            ]))
        print('==> Validation data # frames:',len(validation_set))
        #print validation_set[1]

        if self.for_triplet_loss:
            val_labels = validation_set.get_labels()
            class_qty = len(set(val_labels))

            val_loader = DataLoader(
                dataset=validation_set,
                batch_sampler=BalancedBatchSampler(val_labels, n_classes=class_qty, n_samples=self.batch_size),
                num_workers=self.num_workers)
        else:

            val_loader = DataLoader(
                dataset=validation_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)

        return val_loader
