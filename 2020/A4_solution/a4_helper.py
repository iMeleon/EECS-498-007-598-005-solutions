import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from collections import OrderedDict
import torchvision.datasets as dset
import torchvision.transforms as T

import random
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d

SQUEEZENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float)
SQUEEZENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float)

### Helper Functions
'''
Our pretrained model was trained on images that had been preprocessed by subtracting 
the per-color mean and dividing by the per-color standard deviation. We define a few helper 
functions for performing and undoing this preprocessing.
'''
def preprocess(img, size=224):
  transform = T.Compose([
    T.Resize(size),
    T.ToTensor(),
    T.Normalize(mean=SQUEEZENET_MEAN.tolist(),
          std=SQUEEZENET_STD.tolist()),
    T.Lambda(lambda x: x[None]),
  ])
  return transform(img)

def deprocess(img, should_rescale=True):
  # should_rescale true for style transfer
  transform = T.Compose([
    T.Lambda(lambda x: x[0]),
    T.Normalize(mean=[0, 0, 0], std=(1.0 / SQUEEZENET_STD).tolist()),
    T.Normalize(mean=(-SQUEEZENET_MEAN).tolist(), std=[1, 1, 1]),
    T.Lambda(rescale) if should_rescale else T.Lambda(lambda x: x),
    T.ToPILImage(),
  ])
  return transform(img)

# def deprocess(img):
#     transform = T.Compose([
#         T.Lambda(lambda x: x[0]),
#         T.Normalize(mean=[0, 0, 0], std=[1.0 / s for s in SQUEEZENET_STD.tolist()]),
#         T.Normalize(mean=[-m for m in SQUEEZENET_MEAN.tolist()], std=[1, 1, 1]),
#         T.Lambda(rescale),
#         T.ToPILImage(),
#     ])
#     return transform(img)

def rescale(x):
  low, high = x.min(), x.max()
  x_rescaled = (x - low) / (high - low)
  return x_rescaled
  
def blur_image(X, sigma=1):
  X_np = X.cpu().clone().numpy()
  X_np = gaussian_filter1d(X_np, sigma, axis=2)
  X_np = gaussian_filter1d(X_np, sigma, axis=3)
  X.copy_(torch.Tensor(X_np).type_as(X))
  return X


# Older versions of scipy.misc.imresize yield different results
# from newer versions, so we check to make sure scipy is up to date.
def check_scipy():
    import scipy
    vnum = int(scipy.__version__.split('.')[1])
    major_vnum = int(scipy.__version__.split('.')[0])
    
    assert vnum >= 16 or major_vnum >= 1, "You must install SciPy >= 0.16.0 to complete this notebook."

def jitter(X, ox, oy):
  """
  Helper function to randomly jitter an image.
  
  Inputs
  - X: PyTorch Tensor of shape (N, C, H, W)
  - ox, oy: Integers giving number of pixels to jitter along W and H axes
  
  Returns: A new PyTorch Tensor of shape (N, C, H, W)
  """
  if ox != 0:
    left = X[:, :, :, :-ox]
    right = X[:, :, :, -ox:]
    X = torch.cat([right, left], dim=3)
  if oy != 0:
    top = X[:, :, :-oy]
    bottom = X[:, :, -oy:]
    X = torch.cat([bottom, top], dim=2)
  return X


def load_CIFAR(path='./datasets/'):
  NUM_TRAIN = 49000
  # The torchvision.transforms package provides tools for preprocessing data
  # and for performing data augmentation; here we set up a transform to
  # preprocess the data by subtracting the mean RGB value and dividing by the
  # standard deviation of each RGB value; we've hardcoded the mean and std.
  transform = T.Compose([
                  T.ToTensor(),
                  T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
              ])

  # We set up a Dataset object for each split (train / val / test); Datasets load
  # training examples one at a time, so we wrap each Dataset in a DataLoader which
  # iterates through the Dataset and forms minibatches. We divide the CIFAR-10
  # training set into train and val sets by passing a Sampler object to the
  # DataLoader telling how it should sample from the underlying Dataset.
  cifar10_train = dset.CIFAR10(path, train=True, download=True,
                               transform=transform)
  loader_train = DataLoader(cifar10_train, batch_size=64, 
                            sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

  cifar10_val = dset.CIFAR10(path, train=True, download=True,
                             transform=transform)
  loader_val = DataLoader(cifar10_val, batch_size=64, 
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))

  cifar10_test = dset.CIFAR10(path, train=False, download=True, 
                              transform=transform)
  loader_test = DataLoader(cifar10_test, batch_size=64)
  return loader_train, loader_val, loader_test


def load_imagenet_val(num=None, path='./datasets/imagenet_val_25.npz'):
  """Load a handful of validation images from ImageNet.
  Inputs:
  - num: Number of images to load (max of 25)
  Returns:
  - X: numpy array with shape [num, 224, 224, 3]
  - y: numpy array of integer image labels, shape [num]
  - class_names: dict mapping integer label to class name
  """
  imagenet_fn = os.path.join(path)
  if not os.path.isfile(imagenet_fn):
    print('file %s not found' % imagenet_fn)
    print('Run the above cell to download the data')
    assert False, 'Need to download imagenet_val_25.npz'
  f = np.load(imagenet_fn, allow_pickle=True)
  X = f['X']
  y = f['y']
  class_names = f['label_map'].item()
  if num is not None:
    X = X[:num]
    y = y[:num]
  return X, y, class_names


def load_COCO(path = './datasets/coco.pt'):
  '''
    Download and load serialized COCO data from coco.pt
    It contains a dictionary of
    "train_images" - resized training images (112x112)
    "val_images" - resized validation images (112x112)
    "train_captions" - tokenized and numericalized training captions
    "val_captions" - tokenized and numericalized validation captions
    "vocab" - caption vocabulary, including "idx_to_token" and "token_to_idx"

    Returns: a data dictionary
  '''
  data_dict = torch.load(path)
  # print out all the keys and values from the data dictionary
  for k, v in data_dict.items():
      if type(v) == torch.Tensor:
          print(k, type(v), v.shape, v.dtype)
      else:
          print(k, type(v), v.keys())

  num_train = data_dict['train_images'].size(0)
  num_val = data_dict['val_images'].size(0)
  assert data_dict['train_images'].size(0) == data_dict['train_captions'].size(0) and \
        data_dict['val_images'].size(0) == data_dict['val_captions'].size(0), \
        'shapes of data mismatch!'

  print('\nTrain images shape: ', data_dict['train_images'].shape)
  print('Train caption tokens shape: ', data_dict['train_captions'].shape)
  print('Validation images shape: ', data_dict['val_images'].shape)
  print('Validation caption tokens shape: ', data_dict['val_captions'].shape)
  print('total number of caption tokens: ', len(data_dict['vocab']['idx_to_token']))
  print('mappings (list) from index to caption token: ', data_dict['vocab']['idx_to_token'])
  print('mappings (dict) from caption token to index: ', data_dict['vocab']['token_to_idx'])


  return data_dict


## Dump files for submission
def dump_results(submission, path):
  '''
  Dumps a dictionary as a .pkl file for autograder 
    results: a dictionary 
    path: path for saving the dict object 
  '''
  # del submission['rnn_model']
  # del submission['lstm_model']
  # del submission['attn_model']
  with open(path, "wb") as f:
    pickle.dump(submission, f)









