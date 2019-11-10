"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np



class MLP(nn.Module):
  """
  This class implements a Multi-layer Perceptron in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward.
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

    TODO:
    Implement initialization of the network.
    """

    super(MLP, self).__init__()

    ## setting the layer list as model params
    self.layers = nn.ModuleList()

    input_size = n_inputs
    for layerSize in n_hidden:
      self.layers.append(nn.Linear(input_size,layerSize).cuda())
      self.layers.append(nn.LeakyReLU(neg_slope).cuda())

    ## to dimension of classes
    self.layers.append(nn.Linear(layerSize,n_classes).cuda())
    self.layers.append(nn.LeakyReLU(neg_slope).cuda())

    ## softmax
    self.layers.append(nn.Softmax(dim =  1).cuda())



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

    for layer in self.layers:
      out = layer(out)


    return out
