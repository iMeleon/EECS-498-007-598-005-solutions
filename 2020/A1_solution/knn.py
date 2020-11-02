"""
If this solution was useful to you and you want to thank me, please give me a star:
https://github.com/iMeleon/EECS-498-007-598-005-solutions
"""

"""
Implements a K-Nearest Neighbor classifier in PyTorch.
"""
import torch
import statistics


def hello():
  """
  This is a sample function that we will try to import and run to ensure that
  our environment is correctly set up on Google Colab.
  """
  print('Hello from knn.py!')


def compute_distances_two_loops(x_train, x_test):
  """
  Computes the squared Euclidean distance between each element of the training
  set and each element of the test set. Images should be flattened and treated
  as vectors.

  This implementation uses a naive set of nested loops over the training and
  test data.

  The input data may have any number of dimensions -- for example this function
  should be able to compute nearest neighbor between vectors, in which case
  the inputs will have shape (num_{train, test}, D); it should alse be able to
  compute nearest neighbors between images, where the inputs will have shape
  (num_{train, test}, C, H, W). More generally, the inputs will have shape
  (num_{train, test}, D1, D2, ..., Dn); you should flatten each element
  of shape (D1, D2, ..., Dn) into a vector of shape (D1 * D2 * ... * Dn) before
  computing distances.

  The input tensors should not be modified.

  NOTE: Your implementation may not use `torch.norm`, `torch.dist`,
  `torch.cdist`, or their instance method variants x.norm / x.dist / x.cdist.
  You may not use any functions from torch.nn or torch.nn.functional.

  Inputs:
  - x_train: Torch tensor of shape (num_train, D1, D2, ...)
  - x_test: Torch tensor of shape (num_test, D1, D2, ...)

  Returns:
  - dists: Torch tensor of shape (num_train, num_test) where dists[i, j] is the
    squared Euclidean distance between the ith training point and the jth test
    point. It should have the same dtype as x_train.
  """
  # Initialize dists to be a tensor of shape (num_train, num_test) with the
  # same datatype and device as x_train
  num_train = x_train.shape[0]
  num_test = x_test.shape[0]
  dists = x_train.new_zeros(num_train, num_test)
  ##############################################################################
  # TODO: Implement this function using a pair of nested loops over the        #
  # training data and the test data.                                           #
  #                                                                            #
  # You may not use torch.norm (or its instance method variant), nor any       #
  # functions from torch.nn or torch.nn.functional.                            #
  ##############################################################################
  # Replace "pass" statement with your code
  for i in range(num_train):
    for j in range(num_test):
      dists[i,j] = ((x_train[i] -x_test[j])**2).sum()**(1/2)
  ##############################################################################
  #                             END OF YOUR CODE                               #
  ##############################################################################
  return dists


def compute_distances_one_loop(x_train, x_test):
  """
  Computes the squared Euclidean distance between each element of the training
  set and each element of the test set. Images should be flattened and treated
  as vectors.

  This implementation uses only a single loop over the training data.

  Similar to compute_distances_two_loops, this should be able to handle inputs
  with any number of dimensions. The inputs should not be modified.

  NOTE: Your implementation may not use `torch.norm`, `torch.dist`,
  `torch.cdist`, or their instance method variants x.norm / x.dist / x.cdist.
  You may not use any functions from torch.nn or torch.nn.functional.

  Inputs:
  - x_train: Torch tensor of shape (num_train, D1, D2, ...)
  - x_test: Torch tensor of shape (num_test, D1, D2, ...)

  Returns:
  - dists: Torch tensor of shape (num_train, num_test) where dists[i, j] is the
    squared Euclidean distance between the ith training point and the jth test
    point.
  """
  # Initialize dists to be a tensor of shape (num_train, num_test) with the
  # same datatype and device as x_train
  num_train = x_train.shape[0]
  num_test = x_test.shape[0]
  dists = x_train.new_zeros(num_train, num_test)
  ##############################################################################
  # TODO: Implement this function using only a single loop over x_train.       #
  #                                                                            #
  # You may not use torch.norm (or its instance method variant), nor any       #
  # functions from torch.nn or torch.nn.functional.                            #
  ##############################################################################
  # Replace "pass" statement with your code
  for i in range(num_train):
      dists[i] = ((x_train[i] -x_test)**2).sum(dim  = (1,2,3))**(1/2)
  ##############################################################################
  #                             END OF YOUR CODE                               #
  ##############################################################################
  return dists


def compute_distances_no_loops(x_train, x_test):
  """
  Computes the squared Euclidean distance between each element of the training
  set and each element of the test set. Images should be flattened and treated
  as vectors.

  This implementation should not use any Python loops. For memory-efficiency,
  it also should not create any large intermediate tensors; in particular you
  should not create any intermediate tensors with O(num_train*num_test)
  elements.

  Similar to compute_distances_two_loops, this should be able to handle inputs
  with any number of dimensions. The inputs should not be modified.

  NOTE: Your implementation may not use `torch.norm`, `torch.dist`,
  `torch.cdist`, or their instance method variants x.norm / x.dist / x.cdist.
  You may not use any functions from torch.nn or torch.nn.functional.
  Inputs:
  - x_train: Torch tensor of shape (num_train, C, H, W)
  - x_test: Torch tensor of shape (num_test, C, H, W)

  Returns:
  - dists: Torch tensor of shape (num_train, num_test) where dists[i, j] is the
    squared Euclidean distance between the ith training point and the jth test
    point.
  """
  # Initialize dists to be a tensor of shape (num_train, num_test) with the
  # same datatype and device as x_train
  num_train = x_train.shape[0]
  num_test = x_test.shape[0]
  dists = x_train.new_zeros(num_train, num_test)
  ##############################################################################
  # TODO: Implement this function without using any explicit loops and without #
  # creating any intermediate tensors with O(num_train * num_test) elements.   #
  #                                                                            #
  # You may not use torch.norm (or its instance method variant), nor any       #
  # functions from torch.nn or torch.nn.functional.                            #
  #                                                                            #
  # HINT: Try to formulate the Euclidean distance using two broadcast sums     #
  #       and a matrix multiply.                                               #
  ##############################################################################
  # Replace "pass" statement with your code
  
  # link for understanding :
  # https://stackoverflow.com/questions/27948363/numpy-broadcast-to-perform-euclidean-distance-vectorized
  A = x_train.reshape(num_train,-1)
  B = x_test.reshape(num_test,-1)
  AB2 = A.mm(B.T)*2
  dists = ((A**2).sum(dim = 1).reshape(-1,1) - AB2 + (B**2).sum(dim = 1).reshape(1,-1))**(1/2)
  ##############################################################################
  #                             END OF YOUR CODE                               #
  ##############################################################################
  return dists

def predict_labels(dists, y_train, k=1):
  """
  Given distances between all pairs of training and test samples, predict a
  label for each test sample by taking a **majority vote** among its k nearest
  neighbors in the training set.

  In the event of a tie, this function **should** return the smallest label. For
  example, if k=5 and the 5 nearest neighbors to a test example have labels
  [1, 2, 1, 2, 3] then there is a tie between 1 and 2 (each have 2 votes), so
  we should return 1 since it is the smallest label.
s
  This function should not modify any of its inputs.

  Inputs:
  - dists: Torch tensor of shape (num_train, num_test) where dists[i, j] is the
    squared Euclidean distance between the ith training point and the jth test
    point.
  - y_train: Torch tensor of shape (num_train,) giving labels for all training
    samples. Each label is an integer in the range [0, num_classes - 1]
  - k: The number of nearest neighbors to use for classification.

  Returns:
  - y_pred: A torch int64 tensor of shape (num_test,) giving predicted labels
    for the test data, where y_pred[j] is the predicted label for the jth test
    example. Each label should be an integer in the range [0, num_classes - 1].
  """
  num_train, num_test = dists.shape
  y_pred = torch.zeros(num_test, dtype=torch.int64)
  ##############################################################################
  # TODO: Implement this function. You may use an explicit loop over the test  #
  # samples. Hint: Look up the function torch.topk                             #
  ##############################################################################
  # Replace "pass" statement with your code
  values, indices = torch.topk(dists, k, dim=0, largest=False)
  for i in range(indices.shape[1]):
    _, idx = torch.max(y_train[indices[:,i]].bincount(), dim = 0)
    y_pred[i] = idx
  
  ##############################################################################
  #                             END OF YOUR CODE                               #
  ##############################################################################
  return y_pred


class KnnClassifier:
  def __init__(self, x_train, y_train):
    """
    Create a new K-Nearest Neighbor classifier with the specified training data.
    In the initializer we simply memorize the provided training data.

    Inputs:
    - x_train: Torch tensor of shape (num_train, C, H, W) giving training data
    - y_train: int64 torch tensor of shape (num_train,) giving training labels
    """
    ###########################################################################
    # TODO: Implement the initializer for this class. It should perform no    #
    # computation and simply memorize the training data.                      #
    ###########################################################################
    # Replace "pass" statement with your code
    self.x_train = x_train
    self.y_train = y_train
    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################

  def predict(self, x_test, k=1):
    """
    Make predictions using the classifier.

    Inputs:
    - x_test: Torch tensor of shape (num_test, C, H, W) giving test samples
    - k: The number of neighbors to use for predictions

    Returns:
    - y_test_pred: Torch tensor of shape (num_test,) giving predicted labels
      for the test samples.
    """
    y_test_pred = None
    ###########################################################################
    # TODO: Implement this method. You should use the functions you wrote     #
    # above for computing distances (use the no-loop variant) and to predict  #
    # output labels.
    ###########################################################################
    # Replace "pass" statement with your code
    dists = compute_distances_no_loops(self.x_train, x_test)
    y_test_pred =  predict_labels(dists, self.y_train, k)
    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################
    return y_test_pred

  def check_accuracy(self, x_test, y_test, k=1, quiet=False):
    """
    Utility method for checking the accuracy of this classifier on test data.
    Returns the accuracy of the classifier on the test data, and also prints a
    message giving the accuracy.

    Inputs:
    - x_test: Torch tensor of shape (num_test, C, H, W) giving test samples
    - y_test: int64 torch tensor of shape (num_test,) giving test labels
    - k: The number of neighbors to use for prediction
    - quiet: If True, don't print a message.

    Returns:
    - accuracy: Accuracy of this classifier on the test data, as a percent.
      Python float in the range [0, 100]
    """
    y_test_pred = self.predict(x_test, k=k)
    num_samples = x_test.shape[0]
    num_correct = (y_test == y_test_pred).sum().item()
    accuracy = 100.0 * num_correct / num_samples
    msg = (f'Got {num_correct} / {num_samples} correct; '
           f'accuracy is {accuracy:.2f}%')
    if not quiet:
      print(msg)
    return accuracy


def knn_cross_validate(x_train, y_train, num_folds=5, k_choices=None):
  """
  Perform cross-validation for KnnClassifier.

  Inputs:
  - x_train: Tensor of shape (num_train, C, H, W) giving all training data
  - y_train: int64 tensor of shape (num_train,) giving labels for training data
  - num_folds: Integer giving the number of folds to use
  - k_choices: List of integers giving the values of k to try

  Returns:
  - k_to_accuracies: Dictionary mapping values of k to lists, where
    k_to_accuracies[k][i] is the accuracy on the ith fold of a KnnClassifier
    that uses k nearest neighbors.
  """
  if k_choices is None:
    # Use default values
    k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

  # First we divide the training data into num_folds equally-sized folds.
  x_train_folds = []
  y_train_folds = []
  ##############################################################################
  # TODO: Split the training data and images into folds. After splitting,      #
  # x_train_folds and y_train_folds should be lists of length num_folds, where #
  # y_train_folds[i] is the label vector for images in x_train_folds[i].       #
  # Hint: torch.chunk                                                          #
  ##############################################################################
  # Replace "pass" statement with your code
  x_train_folds = torch.chunk(x_train, num_folds, dim=0)
  y_train_folds = torch.chunk(y_train, num_folds, dim=0)
  ##############################################################################
  #                            END OF YOUR CODE                                #
  ##############################################################################

  # A dictionary holding the accuracies for different values of k that we find
  # when running cross-validation. After running cross-validation,
  # k_to_accuracies[k] should be a list of length num_folds giving the different
  # accuracies we found when trying KnnClassifiers that use k neighbors.
  k_to_accuracies = {}

  ##############################################################################
  # TODO: Perform cross-validation to find the best value of k. For each value #
  # of k in k_choices, run the k-nearest-neighbor algorithm num_folds times;   #
  # in each case you'll use all but one fold as training data, and use the     #
  # last fold as a validation set. Store the accuracies for all folds and all  #
  # values in k in k_to_accuracies.   HINT: torch.cat                          #
  ##############################################################################
  # Replace "pass" statement with your code
  for k in k_choices:
    list_of_acc = []
    for num_fold in range(num_folds):
      x_train_folds_local = [x for x in x_train_folds]
      y_train_folds_local = [x for x in y_train_folds]
      x_test = x_train_folds_local[num_fold]
      y_test = y_train_folds_local[num_fold]
      del x_train_folds_local[num_fold]
      del y_train_folds_local[num_fold]
      x_train = torch.cat(x_train_folds_local, dim=0)
      y_train = torch.cat(y_train_folds_local, dim=0)
      classifier = KnnClassifier(x_train, y_train)
      list_of_acc.append(classifier.check_accuracy(x_test, y_test,k))
    k_to_accuracies[k] = list_of_acc

  ##############################################################################
  #                            END OF YOUR CODE                                #
  ##############################################################################

  return k_to_accuracies


def knn_get_best_k(k_to_accuracies):
  """
  Select the best value for k, from the cross-validation result from
  knn_cross_validate. If there are multiple k's available, then you SHOULD
  choose the smallest k among all possible answer.

  Inputs:
  - k_to_accuracies: Dictionary mapping values of k to lists, where
    k_to_accuracies[k][i] is the accuracy on the ith fold of a KnnClassifier
    that uses k nearest neighbors.

  Returns:
  - best_k: best (and smallest if there is a conflict) k value based on
            the k_to_accuracies info
  """
  best_k = 0
  ##############################################################################
  # TODO: Use the results of cross-validation stored in k_to_accuracies to     #
  # choose the value of k, and store the result in best_k. You should choose   #
  # the value of k that has the highest mean accuracy accross all folds.       #
  ##############################################################################
  # Replace "pass" statement with your code
  new_dict = {}
  for k, accs in sorted(k_to_accuracies.items()):
     new_dict[k] = sum(accs) / len(accs) 
  max_value = max(new_dict.values())
  best_k = [k for k, v in new_dict.items() if v == max_value][0]
  ##############################################################################
  #                            END OF YOUR CODE                                #
  ##############################################################################
  return best_k
"""
If this solution was useful to you and you want to thank me, please give me a star:
https://github.com/iMeleon/EECS-498-007-598-005-solutions
"""