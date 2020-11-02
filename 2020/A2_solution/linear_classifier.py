"""
If this solution was useful to you and you want to thank me, please give me a star:
https://github.com/iMeleon/EECS-498-007-598-005-solutions
"""
"""
Implements linear classifeirs in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
import random
import statistics
from abc import abstractmethod

def hello_linear_classifier():
  """
  This is a sample function that we will try to import and run to ensure that
  our environment is correctly set up on Google Colab.
  """
  print('Hello from linear_classifier.py!')


# Template class modules that we will use later: Do not edit/modify this class
class LinearClassifier(object):
  """ An abstarct class for the linear classifiers """
  # Note: We will re-use `LinearClassifier' in both SVM and Softmax
  def __init__(self):
    random.seed(0)
    torch.manual_seed(0)
    self.W = None

  def train(self, X_train, y_train, learning_rate=1e-3, reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    train_args = (self.loss, self.W, X_train, y_train, learning_rate, reg,
                  num_iters, batch_size, verbose)
    self.W, loss_history = train_linear_classifier(*train_args)
    return loss_history

  def predict(self, X):
    return predict_linear_classifier(self.W, X)

  @abstractmethod
  def loss(self, W, X_batch, y_batch, reg):
    """
    Compute the loss function and its derivative.
    Subclasses will override this.

    Inputs:
    - W: A PyTorch tensor of shape (D, C) containing (trained) weight of a model.
    - X_batch: A PyTorch tensor of shape (N, D) containing a minibatch of N
      data points; each point has dimension D.
    - y_batch: A PyTorch tensor of shape (N,) containing labels for the minibatch.
    - reg: (float) regularization strength.

    Returns: A tuple containing:
    - loss as a single float
    - gradient with respect to self.W; an tensor of the same shape as W
    """
    raise NotImplementedError

  def _loss(self, X_batch, y_batch, reg):
    self.loss(self.W, X_batch, y_batch, reg)

  def save(self, path):
    torch.save({'W': self.W}, path)
    print("Saved in {}".format(path))

  def load(self, path):
    W_dict = torch.load(path, map_location='cpu')
    self.W = W_dict['W']
    print("load checkpoint file: {}".format(path))



class LinearSVM(LinearClassifier):
  """ A subclass that uses the Multiclass SVM loss function """
  def loss(self, W, X_batch, y_batch, reg):
    return svm_loss_vectorized(W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
  """ A subclass that uses the Softmax + Cross-entropy loss function """
  def loss(self, W, X_batch, y_batch, reg):
    return softmax_loss_vectorized(W, X_batch, y_batch, reg)



#**************************************************#
################## Section 1: SVM ##################
#**************************************************#

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples. When you implment the regularization over W, please DO NOT
  multiply the regularization term by 1/2 (no coefficient).

  Inputs:
  - W: A PyTorch tensor of shape (D, C) containing weights.
  - X: A PyTorch tensor of shape (N, D) containing a minibatch of data.
  - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as torch scalar
  - gradient of loss with respect to weights W; a tensor of same shape as W
  """
  dW = torch.zeros_like(W) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  # print(W.t().shape)
  # print(X[0].shape)
  # print(W.t(х).mv(X[0]).shape)
  for i in range(num_train):
    scores = W.t().mv(X[i])
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        #######################################################################
        # TODO:                                                               #
        # Compute the gradient of the loss function and store it dW. (part 1) #
        # Rather than first computing the loss and then computing the         #
        # derivative, it is simple to compute the derivative at the same time #
        # that the loss is being computed.                                    #
        #######################################################################
        # Replace "pass" statement with your code
        dW[:, j] += X[i]  
        dW[:, y[i]]-= X[i]
        #######################################################################
        #                       END OF YOUR CODE                              #
        #######################################################################


  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * torch.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it in dW. (part 2)    #
  #############################################################################
  # Replace "pass" statement with your code
  dW+=2*reg*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation. When you implment
  the regularization over W, please DO NOT multiply the regularization term by
  1/2 (no coefficient). The inputs and outputs are the same as svm_loss_naive.

  Inputs:
  - W: A PyTorch tensor of shape (D, C) containing weights.
  - X: A PyTorch tensor of shape (N, D) containing a minibatch of data.
  - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as torch scalar
  - gradient of loss with respect to weights W; a tensor of same shape as W
  """
  loss = 0.0
  dW = torch.zeros_like(W) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # Replace "pass" statement with your code
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0

  scores = X.mm(W)
  correct_class_score = scores[range(num_train),y]
  margin = scores -  correct_class_score.view(-1,1) + 1 
  margin[range(num_train),y] = 0
  mask = (margin > 0)
  loss = margin[mask].sum()
  
  mask_correct_y = torch.zeros_like(scores,dtype=torch.bool)
  mask_correct_y[range(num_train),y] = True
  margin[margin> 0] = 1
  margin[margin< 0] = 0
  margin[mask_correct_y] = torch.sum(margin, axis = 1)*-1
  dW = margin.T.mm(X).T

 

    
    

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * torch.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it in dW. (part 2)    #
  #############################################################################
  # Replace "pass" statement with your code
  dW+=2*reg*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # Replace "pass" statement with your code
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW


def sample_batch(X, y, num_train, batch_size):
  """
  Sample batch_size elements from the training data and their
  corresponding labels to use in this round of gradient descent.
  """
  X_batch = None
  y_batch = None
  #########################################################################
  # TODO: Store the data in X_batch and their corresponding labels in     #
  # y_batch; after sampling, X_batch should have shape (batch_size, dim)  #
  # and y_batch should have shape (batch_size,)                           #
  #                                                                       #
  # Hint: Use torch.randint to generate indices.                          #
  #########################################################################
  # Replace "pass" statement with your code
  indices = torch.randint(num_train, (batch_size,))
  y_batch = y[indices]
  X_batch = X[indices]
  #########################################################################
  #                       END OF YOUR CODE                                #
  #########################################################################
  return X_batch, y_batch


def train_linear_classifier(loss_func, W, X, y, learning_rate=1e-3,
                            reg=1e-5, num_iters=100, batch_size=200,
                            verbose=False):
  """
  Train this linear classifier using stochastic gradient descent.

  Inputs:
  - loss_func: loss function to use when training. It should take W, X, y
    and reg as input, and output a tuple of (loss, dW)
  - W: A PyTorch tensor of shape (D, C) giving the initial weights of the
    classifier. If W is None then it will be initialized here.
  - X: A PyTorch tensor of shape (N, D) containing training data; there are N
    training samples each of dimension D.
  - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c
    means that X[i] has label 0 <= c < C for C classes.
  - learning_rate: (float) learning rate for optimization.
  - reg: (float) regularization strength.
  - num_iters: (integer) number of steps to take when optimizing
  - batch_size: (integer) number of training examples to use at each step.
  - verbose: (boolean) If true, print progress during optimization.

  Returns: A tuple of:
  - W: The final value of the weight matrix and the end of optimization
  - loss_history: A list of Python scalars giving the values of the loss at each
    training iteration.
  """
  # assume y takes values 0...K-1 where K is number of classes
  num_train, dim = X.shape
  if W is None:
    # lazily initialize W
    num_classes = torch.max(y) + 1
    W = 0.000001 * torch.randn(dim, num_classes, device=X.device, dtype=X.dtype)
  else:
    num_classes = W.shape[1]

  # Run stochastic gradient descent to optimize W
  loss_history = []
  for it in range(num_iters):
    # TODO: implement sample_batch function
    X_batch, y_batch = sample_batch(X, y, num_train, batch_size)

    # evaluate loss and gradient
    loss, grad = loss_func(W, X_batch, y_batch, reg)
    loss_history.append(loss.item())

    # perform parameter update
    #########################################################################
    # TODO:                                                                 #
    # Update the weights using the gradient and the learning rate.          #
    #########################################################################
    # Replace "pass" statement with your code
    W -= learning_rate * grad
    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################
    if verbose and it % 100 == 0:
      print('iteration %d / %d: loss %f' % (it, num_iters, loss))

  return W, loss_history


def predict_linear_classifier(W, X):
  """
  Use the trained weights of this linear classifier to predict labels for
  data points.

  Inputs:
  - W: A PyTorch tensor of shape (D, C), containing weights of a model
  - X: A PyTorch tensor of shape (N, D) containing training data; there are N
    training samples each of dimension D.

  Returns:
  - y_pred: PyTorch int64 tensor of shape (N,) giving predicted labels for each
    elemment of X. Each element of y_pred should be between 0 and C - 1.
  """
  y_pred = torch.zeros(X.shape[0], dtype=torch.int64)
  ###########################################################################
  # TODO:                                                                   #
  # Implement this method. Store the predicted labels in y_pred.            #
  ###########################################################################
  # Replace "pass" statement with your code
  _, y_pred = X.mm(W).max(dim = 1)
  ###########################################################################
  #                           END OF YOUR CODE                              #
  ###########################################################################
  return y_pred

def svm_get_search_params():
  """
  Return candidate hyperparameters for the SVM model. You should provide
  at least two param for each, and total grid search combinations
  should be less than 25.

  Returns:
  - learning_rates: learning rate candidates, e.g. [1e-3, 1e-2, ...]
  - regularization_strengths: regularization strengths candidates
                              e.g. [1e0, 1e1, ...]
  """


  ###########################################################################
  # TODO:   add your own hyper parameter lists.                             #
  ###########################################################################
  # Replace "pass" statement with your code
  learning_rates = [0.000001,0.0001, 0.001, 0.005,0.01,0.05]
  regularization_strengths = [0.001, 0.5,1,3]
  ###########################################################################
  #                           END OF YOUR CODE                              #
  ###########################################################################

  return learning_rates, regularization_strengths


def test_one_param_set(cls, data_dict, lr, reg, num_iters=2000):
  """
  Train a single LinearClassifier instance and return the learned instance
  with train/val accuracy.

  Inputs:
  - cls (LinearClassifier): a newly-created LinearClassifier instance.
                            Train/Validation should perform over this instance
  - data_dict (dict): a dictionary that includes
                      ['X_train', 'y_train', 'X_val', 'y_val']
                      as the keys for training a classifier
  - lr (float): learning rate parameter for training a SVM instance.
  - reg (float): a regularization weight for training a SVM instance.
  - num_iters (int, optional): a number of iterations to train

  Returns:
  - cls (LinearClassifier): a trained LinearClassifier instances with
                            (['X_train', 'y_train'], lr, reg)
                            for num_iter times.
  - train_acc (float): training accuracy of the svm_model
  - val_acc (float): validation accuracy of the svm_model
  """
  train_acc = 0.0 # The accuracy is simply the fraction of data points
  val_acc = 0.0   # that are correctly classified.
  ###########################################################################
  # TODO:                                                                   #
  # Write code that, train a linear SVM on the training set, compute its    #
  # accuracy on the training and validation sets                            #
  #                                                                         #
  # Hint: Once you are confident that your validation code works, you       #
  # should rerun the validation code with the final value for num_iters.    #
  # Before that, please test with small num_iters first                     #
  ###########################################################################
  # Feel free to uncomment this, at the very beginning,
  # and don't forget to remove this line before submitting your final version
  # num_iters = 100

  # Replace "pass" statement with your code
  cls.train(data_dict['X_train'], data_dict['y_train'], lr, reg, num_iters,
            batch_size=200, verbose=False)
  y_train_pred = cls.predict(data_dict['X_train'])
  train_acc = 100.0 * (data_dict['y_train'] == y_train_pred).double().mean().item()

  y_test_pred = cls.predict(data_dict['X_val'])
  val_acc = 100.0 * (data_dict['y_val'] == y_test_pred).double().mean().item()
  ############################################################################
  #                            END OF YOUR CODE                              #
  ############################################################################

  return cls, train_acc, val_acc



#**************************************************#
################ Section 2: Softmax ################
#**************************************************#

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops).  When you implment
  the regularization over W, please DO NOT multiply the regularization term by
  1/2 (no coefficient).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A PyTorch tensor of shape (D, C) containing weights.
  - X: A PyTorch tensor of shape (N, D) containing a minibatch of data.
  - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an tensor of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = torch.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability (Check Numeric Stability #
  # in http://cs231n.github.io/linear-classify/). Plus, don't forget the      #
  # regularization!                                                           #
  #############################################################################
  loss = 0.0
  dw = torch.zeros_like(W)

  num_classes = W.shape[1]
  num_train = X.shape[0]
    
  scores = W.t().mv(X[0])
  correct_class_score = scores[y[0]]


  for i in range(num_train):
    scores = W.t().mv(X[i])
    scores = scores- scores.max()
    correct_class_score = scores[y[i]]
    loss += -correct_class_score  + torch.log(torch.exp(scores).sum())
    for j in range(num_classes):
      if j == y[i]:
        dw[:, j] += torch.exp(scores[j]) / torch.exp(scores).sum() * X[i, :] - X[i, :]
      else:
        dw[:, j] += torch.exp(scores[j]) / torch.exp(scores).sum() * X[i, :]
  
  loss /= num_train
  loss += reg * torch.sum(W * W)

  dw /= num_train
  dw += 2 * reg * W
 
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dw


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.  When you implment the
  regularization over W, please DO NOT multiply the regularization term by 1/2
  (no coefficient).

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = torch.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability (Check Numeric Stability #
  # in http://cs231n.github.io/linear-classify/). Don't forget the            #
  # regularization!                                                           #
  #############################################################################
  # Replace "pass" statement with your code
  # W = W.clone()
  # X = X.clone()
  # y = y.clone()
  # reg = reg
  loss = 0.0
  dw = torch.zeros_like(W)

  num_classes = W.shape[1]
  num_train = X.shape[0]
    

  scores = X.mm(W)
  val, _  = scores.max(dim = 1)
  scores = scores-val.view(-1,1)
  exp_scores = scores.exp()
  exp_scores_sum = exp_scores.sum(dim = 1)
  exp_scores_sum_log = exp_scores_sum.log()
  correct_class_scores = scores[range(num_train), y]
  loss = (exp_scores_sum_log - correct_class_scores).sum()
  zeros = torch.zeros((num_train,num_classes), dtype=torch.float64, device = 'cuda')
  zeros[range(num_train), y] = -1
  minus_X = zeros.t().mm(X)
  dw+= minus_X.t()
  dw+= ((exp_scores/exp_scores_sum.view(-1,1)).t().mm(X)).t()

  loss /= num_train
  loss += reg * torch.sum(W * W)

  dw /= num_train
  dw += 2 * reg * W
 
  return loss, dw



# If this solution was useful to you and you want to thank me, please give me a star:
# https://github.com/iMeleon/EECS-498-007-598-005-solutions


def softmax_get_search_params():
  """
  Return candidate hyperparameters for the Softmax model. You should provide
  at least two param for each, and total grid search combinations
  should be less than 25.

  Returns:
  - learning_rates: learning rate candidates, e.g. [1e-3, 1e-2, ...]
  - regularization_strengths: regularization strengths candidates
                              e.g. [1e0, 1e1, ...]
  """

  ###########################################################################
  # TODO: Add your own hyper parameter lists. This should be similar to the #
  # hyperparameters that you used for the SVM, but you may need to select   #
  # different hyperparameters to achieve good performance with the softmax  #
  # classifier.                                                             #
  ###########################################################################
  # Replace "pass" statement with your code
  learning_rates = np.linspace(1e-2, 2e-1, 10) # [1e-2, 5e-1] 
# regularization strengths candidates
  regularization_strengths = np.linspace(1e-4, 1e-3, 10)
  ###########################################################################
  #                           END OF YOUR CODE                              #
  ###########################################################################

  return learning_rates, regularization_strengths
"""
If this solution was useful to you and you want to thank me, please give me a star:
https://github.com/iMeleon/EECS-498-007-598-005-solutions
"""