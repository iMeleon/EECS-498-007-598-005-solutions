import os
import torch
from torchvision.datasets import CIFAR10


def _extract_tensors(dset, num=None):
  """
  Extract the data and labels from a CIFAR10 dataset object and convert them to
  tensors.

  Input:
  - dset: A torchvision.datasets.CIFAR10 object
  - num: Optional. If provided, the number of samples to keep.

  Returns:
  - x: float32 tensor of shape (N, 3, 32, 32)
  - y: int64 tensor of shape (N,)
  """
  x = torch.tensor(dset.data, dtype=torch.float32).permute(0, 3, 1, 2).div_(255)
  y = torch.tensor(dset.targets, dtype=torch.int64)
  if num is not None:
    if num <= 0 or num > x.shape[0]:
      raise ValueError('Invalid value num=%d; must be in the range [0, %d]'
                       % (num, x.shape[0]))
    x = x[:num].clone()
    y = y[:num].clone()
  return x, y


def cifar10(num_train=None, num_test=None):
  """
  Return the CIFAR10 dataset, automatically downloading it if necessary.
  This function can also subsample the dataset.

  Inputs:
  - num_train: [Optional] How many samples to keep from the training set.
    If not provided, then keep the entire training set.
  - num_test: [Optional] How many samples to keep from the test set.
    If not provided, then keep the entire test set.

  Returns:
  - x_train: float32 tensor of shape (num_train, 3, 32, 32)
  - y_train: int64 tensor of shape (num_train, 3, 32, 32)
  - x_test: float32 tensor of shape (num_test, 3, 32, 32)
  - y_test: int64 tensor of shape (num_test, 3, 32, 32)
  """
  download = not os.path.isdir('cifar-10-batches-py')
  dset_train = CIFAR10(root='.', download=download, train=True)
  dset_test = CIFAR10(root='.', train=False)
  x_train, y_train = _extract_tensors(dset_train, num_train)
  x_test, y_test = _extract_tensors(dset_test, num_test)
 
  return x_train, y_train, x_test, y_test


