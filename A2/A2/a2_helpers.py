"""
Helper functions used in Assignment 2
"""
import torch
import torchvision
import eecs598
import matplotlib.pyplot as plt
import random
import math


def hello_helper():
  """
  This is a sample function that we will try to import and run to ensure that
  our environment is correctly set up on Google Colab.
  """
  print('Hello from a2_helpers.py!')


def get_toy_data(
    num_inputs=5,
    input_size=4,
    hidden_size=10,
    num_classes=3,
    dtype=torch.float32,
    device='cuda'):
  """
  Get toy data for use when developing a two-layer-net.

  Inputs:
  - num_inputs: Integer N giving the data set size
  - input_size: Integer D giving the dimension of input data
  - hidden_size: Integer H giving the number of hidden units in the model
  - num_classes: Integer C giving the number of categories
  - dtype: torch datatype for all returned data
  - device: device on which the output tensors will reside

  Returns a tuple of:
  - toy_X: `dtype` tensor of shape (N, D) giving data points
  - toy_y: int64 tensor of shape (N,) giving labels, where each element is an
    integer in the range [0, C)
  - params: A dictionary of toy model parameters, with keys:
    - 'W1': `dtype` tensor of shape (D, H) giving first-layer weights
    - 'b1': `dtype` tensor of shape (H,) giving first-layer biases
    - 'W2': `dtype` tensor of shape (H, C) giving second-layer weights
    - 'b2': `dtype` tensor of shape (C,) giving second-layer biases
  """
  N = num_inputs
  D = input_size
  H = hidden_size
  C = num_classes

  # We set the random seed for repeatable experiments.
  eecs598.reset_seed(0)

  # Generate some random parameters, storing them in a dict
  params = {}
  params['W1'] = 1e-4 * torch.randn(D, H, device=device, dtype=dtype)
  params['b1'] = torch.zeros(H, device=device, dtype=dtype)
  params['W2'] = 1e-4 * torch.randn(H, C, device=device, dtype=dtype)
  params['b2'] = torch.zeros(C, device=device, dtype=dtype)

  # Generate some random inputs and labels
  toy_X = 10.0 * torch.randn(N, D, device=device, dtype=dtype)
  toy_y = torch.tensor([0, 1, 2, 2, 1], device=device, dtype=torch.int64)

  return toy_X, toy_y, params


################# Visualizations #################

def plot_stats(stat_dict):
  # Plot the loss function and train / validation accuracies
  plt.subplot(1, 2, 1)
  plt.plot(stat_dict['loss_history'], 'o')
  plt.title('Loss history')
  plt.xlabel('Iteration')
  plt.ylabel('Loss')

  plt.subplot(1, 2, 2)
  plt.plot(stat_dict['train_acc_history'], 'o-', label='train')
  plt.plot(stat_dict['val_acc_history'], 'o-', label='val')
  plt.title('Classification accuracy history')
  plt.xlabel('Epoch')
  plt.ylabel('Clasification accuracy')
  plt.legend()

  plt.gcf().set_size_inches(14, 4)
  plt.show()


def visualize_grid(Xs, ubound=255.0, padding=1):
  """
  Reshape a 4D tensor of image data to a grid for easy visualization.

  Inputs:
  - Xs: Data of shape (N, H, W, C)
  - ubound: Output grid will have values scaled to the range [0, ubound]
  - padding: The number of blank pixels between elements of the grid
  """
  (N, H, W, C) = Xs.shape
  # print(Xs.shape)
  grid_size = int(math.ceil(math.sqrt(N)))
  grid_height = H * grid_size + padding * (grid_size - 1)
  grid_width = W * grid_size + padding * (grid_size - 1)
  grid = torch.zeros((grid_height, grid_width, C), device=Xs.device)
  next_idx = 0
  y0, y1 = 0, H
  for y in range(grid_size):
    x0, x1 = 0, W
    for x in range(grid_size):
      if next_idx < N:
        img = Xs[next_idx]
        low, high = torch.min(img), torch.max(img)
        grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
        next_idx += 1
      x0 += W + padding
      x1 += W + padding
    y0 += H + padding
    y1 += H + padding
  return grid


# Visualize the weights of the network
def show_net_weights(net):
  W1 = net.params['W1']
  W1 = W1.reshape(3, 32, 32, -1).transpose(0, 3)
  plt.imshow(visualize_grid(W1, padding=3).type(torch.uint8).cpu())
  plt.gca().axis('off')
  plt.show()


def plot_acc_curves(stat_dict):
  plt.subplot(1, 2, 1)
  for key, single_stats in stat_dict.items():
    plt.plot(single_stats['train_acc_history'], label=str(key))
  plt.title('Train accuracy history')
  plt.xlabel('Epoch')
  plt.ylabel('Clasification accuracy')

  plt.subplot(1, 2, 2)
  for key, single_stats in stat_dict.items():
    plt.plot(single_stats['val_acc_history'], label=str(key))
  plt.title('Validation accuracy history')
  plt.xlabel('Epoch')
  plt.ylabel('Clasification accuracy')
  plt.legend()

  plt.gcf().set_size_inches(14, 5)
  plt.show()
