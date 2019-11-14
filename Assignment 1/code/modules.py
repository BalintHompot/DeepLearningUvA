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

    out = np.dot( x, np.transpose(self.params['weight'])) + self.params['bias']
    self.lastActivity = out
    self.lastInput = x

    return out

  def backward(self, dout):
    batchSize = np.shape(dout)[0]
    dx = np.dot(dout, self.derivative() )

    ## weight gradients - storing for update at the end of batch, called by separate function

    weigth_grads = np.dot(np.transpose(dout), self.lastInput)
    self.params['weight'] += self.learningRate * (weigth_grads/batchSize)

    ## bias - derivative of activity is 1 w.r.t. to bias => 1*dout
    self.params['bias'] += self.learningRate * (np.sum(dout, axis=0)/batchSize)
    return dx

  def derivative(self):
    return self.params['weight']


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
  

class SoftMaxModule(object):
  """
  Softmax activation module.
  """

  def forward(self, x):
 
    b = np.max(x, 1)
    sub = np.transpose(np.subtract(np.transpose(x) , b))
    out = np.exp(sub)
    out = np.transpose(np.divide(np.transpose(out), np.sum(out, 1)))
    self.lastActivity = out

    return out

  def backward(self, dout):

    d = self.derivative(self.lastActivity)
    dx = []
    for instance in range(np.shape(d)[0]):
      dx.append(np.dot(np.transpose(d[instance]), dout[instance]))

    return dx

  def derivative(self, x):
    
    d = []
    for instanceInd in range(len(x)):
      d.append([])
      for ind in range(len(x[instanceInd])):
        d[instanceInd].append([])
        for act in range(len(x[instanceInd])):
          if ind == act:
            d[instanceInd][ind].append((x[instanceInd][ind]*(1-x[instanceInd][act])))
          else:
            d[instanceInd][ind].append((x[instanceInd][ind]*(-x[instanceInd][act])))
        
    return d


class CrossEntropyModule(object):

  def forward(self, x, y):


    out = - np.dot(np.transpose(x), y)
    self.lastActivity = out

    return out

  def backward(self, x, y):

    x[x<=0.00001] = 0.00001
    dx = np.divide(y  , x )
    return dx
  