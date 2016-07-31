import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    esum = np.sum(np.exp(scores))
    logesum = np.log(np.sum(np.exp(scores)))
    loss += -correct_class_score + logesum
    for j in xrange(num_classes):
      dW[:,j] += (1/esum) * X[i]*np.exp(scores[j])
      if j == y[i]:
        dW[:,j] -= X[i]


  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  dW /= num_train
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  num_train = X.shape[0]
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W) #(N, C)
  esums = np.sum(np.exp(scores), axis=1)
  loss -= np.sum(scores[range(y.shape[0]), y])
  loss += np.sum(np.log(esums))
  loss /= num_train
  loss += 0.5*reg*np.sum(W*W)

  dW += X.T.dot(np.exp(scores)*(1/esums.reshape(500, 1)))
  for i in range(y.shape[0]):
    dW[:, y[i]] -= X[i, :]
  dW /= num_train
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

