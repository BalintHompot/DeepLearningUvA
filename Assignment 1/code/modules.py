"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data. 
  """
  def __init__(self, in_features, out_features, learningRate):

    ##init values
    self.in_features = in_features
    self.out_features = out_features
    weights = np.random.normal(0, 0.0001,(out_features, in_features))
    biases = np.random.normal(0,0.0001, out_features)

    gradsW = np.zeros((out_features, in_features))
    gradsB = np.zeros(out_features)
    
    ##storing
    self.params = {'weight': weights, 'bias': biases}
    self.grads = {'weight': gradsW, 'bias': gradsB}
    self.learningRate = learningRate

    ### storing for backward pass
    self.lastActivity = []

  def forward(self, x):

    out = np.dot(self.params['weight'], x) + self.params['bias']
    self.lastActivity = out
    self.lastInput = x

    return out

  def backward(self, dout):
    dx = np.dot(np.transpose(self.derivative()), dout)

    ## weight gradients - storing for update at the end of batch, called by separate function
    weigth_grads = np.outer(dout, self.lastInput)
    self.grads['weight'] += weigth_grads

    ## bias - derivative of activity is 1 w.r.t. to bias => 1*dout
    self.grads['bias'] += dout
    return dx

  def derivative(self):
    return self.params['weight']

  def update(self,batchSize):

    self.params['weight'] -= self.learningRate * self.grads['weight']/batchSize
    self.params['bias'] -= self.learningRate * self.grads['bias']/batchSize

    ## resetting
    gradsW = np.zeros((self.out_features, self.in_features))
    gradsB = np.zeros(self.out_features)
    self.grads = {'weight': gradsW, 'bias': gradsB}

class LeakyReLUModule(object):
  """
  Leaky ReLU activation module.
  """
  def __init__(self, neg_slope):


    self.neg_slope = neg_slope
    self.lastActivity = []

  def forward(self, x):

    out = np.where(x>0, x, x*self.neg_slope)
    self.lastActivity = out

    return out

  def backward(self, dout):

    dx = np.multiply(self.derivative(self.lastActivity) , dout ) 

    return dx

  def derivative(self,x):
    return np.where(x>0, 1, self.neg_slope)
  
  def update(self,batchSize):
    ## we don't update the relu weights
    pass

class SoftMaxModule(object):
  """
  Softmax activation module.
  """

  def forward(self, x):

    x = np.nan_to_num(x)
    b = x.max()

    out = np.exp(x - b)

    out = out / out.sum()
    self.lastActivity = out

    return out

  def backward(self, dout):

    dx = np.dot(self.derivative(self.lastActivity), dout)

    return dx

  def derivative(self, x):
    
    d = []
    for ind in range(len(x)):
      d.append([])
      for act in range(len(x)):
        if ind == act:
          d[ind].append(x[ind]*(1-x[act]))
        else:
          d[ind].append(x[ind]*(-x[act]))
      
    return d

  def update(self,batchSize):
    ## we don't update the softmax weights
    pass

class CrossEntropyModule(object):

  def forward(self, x, y):


    out = - np.dot(np.transpose(x), y)
    self.lastActivity = out

    return out

  def backward(self, x, y):

    ## smoothing for 0 output division
    x[x==0] = 0.000000000001
    dx = np.divide(y, x)

    return dx
  
  def update(self,batchSize):
    ## we don't update the crossentropy weights
    pass