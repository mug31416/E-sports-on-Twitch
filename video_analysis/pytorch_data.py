#!/usr/bin/env python3

import os, gc, time, random
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image

LABEL_DICT = {'0' : 0, '1' : 1}
INVERSE_LABEL_DICT = dict( [(val, key) for key, val in LABEL_DICT.items()] )
CLASS_QTY=len(LABEL_DICT)

def my_initializer(m):

  if hasattr(m, 'weight'):
    torch.nn.init.xavier_uniform_(m.weight.data)
  if hasattr(m, 'bias') and m.bias is not None:
    print('Zero initing bias')
    m.bias.data.zero_()

MIN_WIDTH_HEIGHT=224
 
def resize_img(img, width, height, min_width_height):

  if width < height:
    target_size_width = min_width_height
    target_size_height = int(height * min_width_height / width) # should be >= min_width_height b/c height > width
  else:
    target_size_width = int(width * min_width_height / height)
    target_size_height = min_width_height

 
  assert(target_size_height >= min_width_height)
  assert(target_size_width >= min_width_height)
  
  img = cv2.resize(img, (target_size_width, target_size_height), interpolation = cv2.INTER_CUBIC)

  return img, target_size_width, target_size_height
   
 

def get_labels(label_file):

  line_num = 0
  with open(label_file) as f: 
    for line in f:
      line_num += 1
      line = line.strip()
      if not line:
        continue
      fields = line.split()
  
      if len(fields) != 2:
        raise Exception(f'Wrong format line {line_num} file {label_file}')
  
      file_name_prefix, label_str = fields
  
      if label_str not in LABEL_DICT:
        raise Exception(f'Wrong label {label_str} file {label_file}')
  
      lab = LABEL_DICT[label_str]

      yield line_num, file_name_prefix, label_str


class GetFrames:

  def __init__(self, video_filename):

    self.video_filename = video_filename

    # Create video capture object
    self.vcap = cv2.VideoCapture(video_filename)

    self.width = self.vcap.get(cv2.CAP_PROP_FRAME_WIDTH)
    self.height = self.vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    self.fps = self.vcap.get(cv2.CAP_PROP_FPS)

    self.frame = 0
    self.img = None

    print('File %s width %d height %d FPS %g' % (video_filename, self.width, self.height, self.fps))


  def read_next(self):
    self.frame += 1
    ret, self.img = self.vcap.read()
    return ret

  def release(self):
    print('Releasing vcap for %s last read frame %d' % (self.video_filename, self.frame))
    self.vcap.release()
  

class VideoFramesBase(data.Dataset):

  def __init__(self):

      # With respect to image ingestion to ResNet
      # we follow this example: 
      # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L198
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    self.pytorch_img_transform = transforms.Compose([transforms.RandomResizedCrop(MIN_WIDTH_HEIGHT),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      normalize])

  def get_image_tensor(self, img, width, height):

    # get_image_tensor was refactored (the resizing part), but it wasn't tested after refactoring
    img, target_size_width, target_size_height  = resize_img(img, width, height, MIN_WIDTH_HEIGHT)
    print('Image target vector dimensions:', target_size_width, target_size_height)
        
    
    # Torch vision transform tools read tensors from PIL (by default)
    # Hence Image.fromarray transformation 
            
    tran_img_tensor = self.pytorch_img_transform(Image.fromarray(img))
    return tran_img_tensor

class VideoFramesTest(VideoFramesBase):
  
  def cache_file_name(cache_file_prefix, file_name_prefix):
  
    return os.path.join(cache_file_prefix, file_name_prefix + '.npy')

  def __init__(self, label_file, file_dir, frame_qty, keyframe_skip, keyframe_interval, cache_file_prefix, file_ext = 'mp4'):

    VideoFramesBase.__init__(self)

    for line_num, file_name_prefix, label_str in get_labels(label_file):

      lab = LABEL_DICT[label_str]

      cache_file_data = VideoFramesTest.cache_file_name(cache_file_prefix, file_name_prefix)
      if os.path.exists(cache_file_data):
        print(f'File {cache_file_data} is already generated!')
        continue
  
      gc.collect()
      file_name = os.path.join(file_dir, file_name_prefix + '.' + file_ext) 
      tensor_list = []

      statinfo = os.stat(file_name)
      if False and statinfo.st_size == 0:
        print(f'Empty file {file_name}, generating zero tensor')
        tensor_list.append(torch.zeros(3, MIN_WIDTH_HEIGHT, MIN_WIDTH_HEIGHT))
      else:

        frame_iter = GetFrames(file_name)
  
        while frame_iter.read_next():
        
          frame = frame_iter.frame
          if frame >= keyframe_skip:

            if (frame - keyframe_skip) % keyframe_interval != 0:
              continue

            tran_img_tensor = self.get_image_tensor(frame_iter.img, frame_iter.width, frame_iter.height)
            tensor_list.append(tran_img_tensor)

            if len(tensor_list) >= frame_qty:
              break
  
        frame_iter.release()


      if not tensor_list:
        raise Exception(f'Faulty video {file_name}, remove it from the list!')

      data = torch.stack(tensor_list)
      frame_qty = len(tensor_list)

      print(f'Saving {frame_qty} frames {cache_file_data}')
      np.save(cache_file_data, data.numpy())


class VideoFramesTrain(VideoFramesBase):

  def cache_file_names(cache_file_prefix, keyframe_interval, subset_id):
  
    if cache_file_prefix is None:
      return None, None
    else:
      add = f'_{keyframe_interval}_{subset_id}' 
      return cache_file_prefix + add + '_data.npy', cache_file_prefix + add + '_lab.npy'

  def __init__(self, label_file, file_dir, keyframe_interval, subset_id, cache_file_prefix=None, file_ext = 'mp4'):

    VideoFramesBase.__init__(self)

    cache_file_data, cache_file_lab = VideoFramesTrain.cache_file_names(cache_file_prefix, keyframe_interval, subset_id)

    if cache_file_data is None or (not os.path.exists(cache_file_lab)) or (not os.path.exists(cache_file_data)):

      self.keyframe_interval = keyframe_interval
      self.subset_id = subset_id


      label_arr = [] 
      image_tensor_arr = []

      for line_num, file_name_prefix, label_str in get_labels(label_file):

        lab = LABEL_DICT[label_str]
  
        gc.collect()
        file_name = os.path.join(file_dir, file_name_prefix + '.' + file_ext) 

        frame_iter = GetFrames(file_name)

        while frame_iter.read_next():

          frame = frame_iter.frame

          if random.randint(0, keyframe_interval - 1) != 0:
            continue
  
          print(f'File {file_name} Image source vector dimensions: {frame_iter.width}x{frame_iter.height}')

          label_arr.append(torch.IntTensor([lab]))

          tran_img_tensor = self.get_image_tensor(frame_iter.img, frame_iter.width, frame_iter.height)
            
          image_tensor_arr.append(tran_img_tensor)
  
          print('Resized image shape:', frame_iter.img.shape, 'Transformed image shape:', image_tensor_arr[-1].size())

        frame_iter.release()
        
  
      self.labs = torch.stack(label_arr)
      self.data = torch.stack(image_tensor_arr)

      if cache_file_data is not None:

        print(f'Caching data to {cache_file_data} and {cache_file_lab}')

        np.save(cache_file_data, self.data.numpy())
        np.save(cache_file_lab, self.labs.numpy())

    else:

      print('Loading cached data!')
      self.data = torch.FloatTensor(np.load(cache_file_data))
      self.labs = torch.IntTensor(np.load(cache_file_lab))

    self.data = self.data.float()
    self.labs = self.labs.long()

    print('Data dimensions:', self.data.size())

  def get_labs(self):
    return self.labs.cpu().numpy()


  def get_class_weights(self):

    freq = np.zeros(CLASS_QTY)

    cpu_labs = self.labs.cpu().numpy()

    for c in cpu_labs:
      freq[c] += 1

    class_wght = 1 / (freq + 1e-6)

    return class_wght


  # One per data point
  def get_weights(self):

    class_wght = self.get_class_weights()

    cpu_labs = self.labs.cpu().numpy()

    return np.array([class_wght[y] for y in cpu_labs])

  def __getitem__(self, index):
    """
    :param index: Index or slice object
    :returns tuple: (y ,x) where target is index of the target class.
    """
    x, y = self.data[index], self.labs[index]

    return x, y

  def __len__(self):

    return self.data.size(0)
