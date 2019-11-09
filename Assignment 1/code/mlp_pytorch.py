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
    self.layers = []
    self.relus = []
    input_size = n_inputs
    for layerSize in n_hidden:
      self.layers.append(nn.Linear(input_size,layerSize))
      self.relus.append(nn.ReLU(layerSize))

    ## to dimension of classes
    self.linear_final = nn.Linear(layerSize,n_classes)
    self.relu_final = nn.ReLU(n_classes)

    ## softmax
    self.softmax = nn.Softmax(dim =  2)


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
    for layerInd in range(len(self.layers)):
      out = self.layers[layerInd](out)
      out = self.relus[layerInd](out)

    out = self.relu_final(self.linear_final(out))
    out = self.softmax(out)
    return out

dummydata = [[1,2,3,1,2], [2,3,4,5,6], [1,2,5,6,2]]
dummyLabels = [1,2,0]
mlp = MLP(5,[5,4], 3, 0.01)

import torch.optim as optim

def criterion(out, label):
    return (label - out)**2

optimizer = optim.SGD(mlp.parameters(), lr=0.001)

for epoch in range(1000):
    X, Y = Variable(torch.Tensor([dummydata])), Variable(torch.Tensor([dummyLabels]))
    optimizer.zero_grad()
    criterion = nn.CrossEntropyLoss()
    outputs = mlp(X)
    #print("outputs")
    #print(outputs)
    #print("labels")
    #print(Y)
    loss = criterion(outputs, Y.long())
    loss.backward()
    optimizer.step()
    print("Epoch {} - loss: {}".format(epoch, loss.data))