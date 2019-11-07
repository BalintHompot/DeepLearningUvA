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
    """
    Initializes the parameters of the module. 
    
    Args:
      in_features: size of each input sample
      out_features: size of each output sample

      ###
      added learning rate as param

    """

    ##init values
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
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    out = np.dot(self.params['weight'], x) + self.params['bias']
    self.lastActivity = out
    self.lastInput = x

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module. Store gradient of the loss with respect to 
    layer parameters in self.grads['weight'] and self.grads['bias']. 
    """

    upper_grads_w = np.dot(np.transpose(self.params['weight']), dout)
    dx = np.multiply(self.derivative(self.lastInput) ,upper_grads_w)
 

    ## weight update - adding, as we are minimizing
    weigth_grads = np.outer(dout, self.lastInput)
    self.params['weight'] += self.learningRate*weigth_grads

    ## bias - derivative of activity is 1 w.r.t. to bias => 1*dout
    self.params['bias'] += self.learningRate*dout

    return dx

  def derivative(self, x):
    return x

class LeakyReLUModule(object):
  """
  Leaky ReLU activation module.
  """
  def __init__(self, neg_slope):
    """
    Initializes the parameters of the module.

    Args:
      neg_slope: negative slope parameter.

    TODO:
    Initialize the module.
    """

    self.neg_slope = neg_slope
    self.lastActivity = []

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    out = np.where(x>0, x, x*self.neg_slope)
    self.lastActivity = out

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """
    dx = np.multiply(self.derivative(self.lastActivity) , dout ) 

    return dx

  def derivative(self,x):
    return np.where(x>0, 1, self.neg_slope)

class SoftMaxModule(object):
  """
  Softmax activation module.
  """

  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module

    TODO:
    Implement forward pass of the module.
    To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
    """

    b = x.max()

    out = np.exp(x - b)

    out = out / out.sum()
    self.lastActivity = out

    return out

  def backward(self, dout):
    """
    Backward pass.
    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module

    TODO:
    Implement backward pass of the module.
    """

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

class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """

  def forward(self, x, y):
    """
    Forward pass.
    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss

    TODO:
    Implement forward pass of the module.
    """

    out = - np.dot(np.transpose(x), y)
    self.lastActivity = out

    return out

  def backward(self, x, y):
    """
    Backward pass.
    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.

    TODO:
    Implement backward pass of the module.
    """

    dx = np.divide(y, x)

    return dx