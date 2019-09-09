import numpy as np
import pickle, os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
from skimage import io, color, exposure
from .triplet_loss import BalancedBatchSampler

#Largely based on the code from https://github.com/jeffreyhuang1/two-stream-action-recognition

class spatial_dataset(Dataset):  
    # Totally dislike this organization, but to keep compatible with the previous code.
    # dic.values() are always labels
    # dic.keys() are tuples: (<video-prefix-name>, <number>)
    # 1. In the training regime, the number is the total number of frames,
    #    each call to getitem selects several frames randomly and
    #    returns them in the form of a dictionary.
    # 2. In the validation regime, the number is the ID of the frame to return.
    #    Thus, getitem would return always the same frame (unlike training)
    def __init__(self, dic, root_dir, mode, transform=None):
 
        self.keys = list(dic.keys())
        self.values = list(dic.values())
        self.root_dir = root_dir
        self.mode =mode
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def load_ucf_image(self, video_name, idx):

        frame_idx = 'frame'+ str(idx).zfill(6)
        img = Image.open(os.path.join(self.root_dir, video_name, frame_idx + '.jpg'))
        transformed_img = self.transform(img)
        img.close()

        return transformed_img

    def get_labels(self):
        return np.array([int(v) for v in self.values])

    def __getitem__(self, idx):

        if self.mode == 'train':
            video_name, nb_clips = self.keys[idx]
            nb_clips = int(nb_clips)
            clips = []
            clips.append(random.randint(1, int(nb_clips/3)))
            clips.append(random.randint(int(nb_clips/3), int(nb_clips*2/3)))
            clips.append(random.randint(int(nb_clips*2/3), int(nb_clips+1)))
            
        elif self.mode == 'val':
            video_name, index = self.keys[idx]
            index =abs(int(index))
        else:
            raise ValueError('There are only train and val mode')

        label = self.values[idx]
        label = int(label)
        
        if self.mode=='train':
            data ={}
            for i in range(len(clips)):
                key = 'img'+str(i)
                index = clips[i]
                data[key] = self.load_ucf_image(video_name, index)
                    
            sample = (data, label)
        elif self.mode=='val':
            data = self.load_ucf_image(video_name,index)
            sample = (video_name, data, label)
        else:
            raise ValueError('There are only train and val mode')
           
        return sample

class Spatial_Dataloader:
    def __init__(self, is_val, batch_size, num_workers, root_dir, desc_numpy_file,
                 val_sample_qty=3, for_triplet_loss=False):

        self.batch_size=batch_size
        self.num_workers=num_workers
        self.data_path = root_dir
        self.frame_count ={}
        # frame_count is a dictionary mapping video file (pattern) to the number of video frames
        self.frame_count={}
        # all_videos is a dictionary mapping video file (pattern) to the label
        self.all_videos = {}
        self.for_triplet_loss = for_triplet_loss

        for label, file_name_pat, frame_count in np.load(desc_numpy_file):
             self.frame_count[file_name_pat] = int(frame_count) 
             self.all_videos[file_name_pat] = int(label) 
        
        if is_val:
            self.loader = self.val(val_sample_qty)
        else:
            self.loader = self.train()
        
    def get_loader(self):
        return self.loader    

    def build_training_dict(self):
        #print '==> Generate frame numbers of each training video'
        self.video_dict={}
        for video in self.all_videos:
            #print videoname
            nb_frame = self.frame_count[video]-10+1
            key = (video,nb_frame)
            self.video_dict[key] = self.all_videos[video]
                    
    def build_val_dict(self, val_sample_qty):
        self.video_dict = {}
        
        for video in self.all_videos:
            sampling_interval = int((self.frame_count[video]-10+1)/val_sample_qty)
            for index in range(val_sample_qty):
                clip_idx = index*sampling_interval
                key = (video, clip_idx+1)
                self.video_dict[key] = self.all_videos[video]

    def train(self):
        self.build_training_dict()
        training_set = spatial_dataset(dic=self.video_dict, root_dir=self.data_path, mode='train', transform = transforms.Compose([
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))
        print('==> Training data :',len(training_set),'frames')

        if self.for_triplet_loss:

            train_labels = training_set.get_labels()
            class_qty = len(set(train_labels))

            train_loader = DataLoader(
                dataset=training_set,
                batch_sampler=BalancedBatchSampler(train_labels, n_classes=class_qty, n_samples=self.batch_size),
                num_workers=self.num_workers
                )

        else:

            train_loader = DataLoader(
                dataset=training_set,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)

        return train_loader

    def val(self, val_sample_qty):
        self.build_val_dict(val_sample_qty)
        validation_set = spatial_dataset(dic=self.video_dict, root_dir=self.data_path, mode='val', transform = transforms.Compose([
                transforms.Scale([224,224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))
        
        print('==> Validation data :',len(validation_set),'frames')

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




