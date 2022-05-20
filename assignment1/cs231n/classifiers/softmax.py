from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
 
    score = np.dot(X,W)
    N = X.shape[0]
    C = 10

    for i in range(N):
      curscore = score[i]
      unnormalized = np.exp(score[i])
      normalize = unnormalized/sum(np.exp(score[i]))
      loss = loss -(np.log(normalize[y[i]]))

      for j in range(C):
        dW[:,j] += X[i]*normalize[j]
      dW[:,y[i]] -= X[i]

    dW = dW/N
    dW += reg*2*W 

    loss = loss/N
    loss += reg*np.sum(np.square(W))
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N = X.shape[0]
    C = W.shape[1]
    score = np.dot(X,W)

    normalize = np.exp(score)
    unnormalize = normalize/normalize.sum(axis = 1, keepdims = True)
    lossarray = normalize[np.arange(N),y]
    loss = np.sum(-np.log(lossarray))/N + reg*np.sum(np.square(W))

    unnormalize[np.arange(N),y] -= 1
    dW = X.T.dot(unnormalize)

    dW = dW/N + reg*2*W


    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
