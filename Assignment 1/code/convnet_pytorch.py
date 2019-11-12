"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

## custom flatten layer for simpler architecture
class Flatten(nn.Module):
    def __init__(self, size):
      super(Flatten, self).__init__()
      self.size = size

    def forward(self, x):        
      x = x.view(-1, self.size)
      return x

class ConvNet(nn.Module):
  """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """

  def __init__(self, n_channels, n_classes):
    """
    Initializes ConvNet object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
                 
    
    TODO:
    Implement initialization of the network.
    """
    super(ConvNet, self).__init__()

    ## setting the layer list as model params
    self.layers = nn.ModuleList()

    
    self.layers.append(nn.Conv2d(n_channels, 64, 3, 1, 1).cuda())
    self.layers.append(nn.ReLU().cuda())
    self.layers.append(nn.MaxPool2d(3, 2, 1).cuda())   

    self.layers.append(nn.Conv2d(64, 128, 3, 1, 1).cuda())
    self.layers.append(nn.ReLU().cuda())
    self.layers.append(nn.MaxPool2d(3, 2, 1).cuda())  

    self.layers.append(nn.Conv2d(128, 256, 3, 1, 1).cuda())
    self.layers.append(nn.ReLU().cuda())
    self.layers.append(nn.Conv2d(256, 256, 3, 1, 1).cuda())
    self.layers.append(nn.ReLU().cuda())
    self.layers.append(nn.MaxPool2d(3, 2, 1).cuda())  

    self.layers.append(nn.Conv2d(256, 512, 3, 1, 1).cuda())
    self.layers.append(nn.ReLU().cuda())
    self.layers.append(nn.Conv2d(512, 512, 3, 1, 1).cuda())
    self.layers.append(nn.ReLU().cuda())
    self.layers.append(nn.MaxPool2d(3, 2, 1).cuda())  

    self.layers.append(nn.Conv2d(512, 512, 3, 1, 1).cuda())
    self.layers.append(nn.ReLU().cuda())
    self.layers.append(nn.Conv2d(512, 512, 3, 1, 1).cuda())
    self.layers.append(nn.ReLU().cuda())
    self.layers.append(nn.MaxPool2d(3, 2, 1).cuda())  

    ## fully connected
    self.layers.append(Flatten(512).cuda())
    self.layers.append(nn.Linear(512,n_classes).cuda())
    self.layers.append(nn.ReLU().cuda())
    


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
