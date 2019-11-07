"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import * 

class MLP(object):
  """
  This class implements a Multi-layer Perceptron in NumPy.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward and backward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes, neg_slope):
    """
    Initializes MLP object. 
    
    Args:
      n_inputs: number of inputs.
      n_hidden: list of ints, specifies the number of units
                in each linear layer. If the list is empty, the MLP
                will not have any linear layers, and the model
                will simply perform a multinomial logistic regression.
      n_classes: number of classes of the classification problem.
                 This number is required in order to specify the
                 output dimensions of the MLP
      neg_slope: negative slope parameter for LeakyReLU
    """
    self.n_inputs = n_inputs
    self.n_hidden = n_hidden
    self.n_classes = n_classes
    self.neg_slope = neg_slope
    self.modules = []

    #Init architecture
    ################# Hidden cycles ##################
    in_size = n_inputs
    for hidden_size in n_hidden:
      #Linear
      self.modules.append(LinearModule(in_size, hidden_size, 0.001))
      #ReLU
      self.modules.append(LeakyReLUModule(neg_slope))
      in_size = hidden_size
    
    #Output: softmax
    self.modules.append(SoftMaxModule())
    #CrossEntropyLoss
    self.loss = CrossEntropyModule()

    print('----------MLP initialized-------------')

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """
    out = x
    for module in self.modules:
      out = module.forward(out)
    return out


  def backward(self, dout):
    """
    Performs backward pass given the gradients of the loss. 

    Args:
      dout: gradients of the loss
    
    TODO:
    Implement backward pass of the network.
    """
    
    grad = dout
    for module in self.modules[::-1]:
      grad = module.backward(grad)

    return


target = [0,1,0]
mlp = MLP(5, [2,4,10,3], 10, 0.1)
for j in range(0,1000000):
  out = mlp.forward([1,2,3,5,5])
  print("output is")
  print(out)
  loss = mlp.loss.forward(out, target)
  lossGr = mlp.loss.backward(out, target)

  mlp.backward(lossGr)