"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from accuracies import drawPlot, accuracy

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from matplotlib import pyplot as plt

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 50
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def calculateTest(test_data_cpu, test_labels_cpu, numBatches, classifier, criterion = None, printing = False, lossCalc = True, test_labels_onehot = None):
  
  startInd = 0
  batchLen = int(np.shape(test_data_cpu)[0]/numBatches)
  endInd = batchLen
  test_loss = 0
  total_loss = 0
  total_acc = 0

  for batch in range(numBatches):
    X_test = test_data_cpu[startInd:endInd]
    if torch.cuda.is_available():
      X_test = X_test.cuda()
    test_output = classifier(X_test)
    test_out_np = test_output.cpu().detach().numpy()
    if lossCalc:
      Y_test = test_labels_onehot[startInd:endInd]
      if torch.cuda.is_available():
        Y_test = Y_test.cuda()
      test_loss = criterion(test_output, Y_test.long())
    test_acc = accuracy(test_out_np, test_labels_cpu[startInd:endInd])
    startInd = endInd
    endInd += batchLen
    total_acc += test_acc
    total_loss += test_loss

    X_test.detach()
    Y_test.detach()


  return total_acc/numBatches, total_loss/numBatches

    

def train():
  """
  Performs training and evaluation of ConvNet model. 

  TODO:
  Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  cifar10 = cifar10_utils.get_cifar10("./cifar10/cifar-10-batches-py")
  training_set = cifar10['train']
  test_set = cifar10['test']
  f = vars(FLAGS)
  input_size = 3*32*32
  number_of_classes = 10
  number_of_channels = 3
  batch_size = f['batch_size']
  ### definition of architecture:
  cnn = ConvNet(number_of_channels, number_of_classes)

  lastEpochNum = 0
  batchCounter = 0
  epoch_acc = 0
  epoch_loss = 0

  optimizer = optim.Adam(cnn.parameters(), lr=f['learning_rate'])
  criterion = torch.nn.CrossEntropyLoss()

  ## preparing test data
  print(np.shape(test_set.images))
  test_data, test_labels = test_set.images[0:200], test_set.labels[0:200]

  ### normalize
  test_data = np.subtract(test_data,np.mean(test_data, 0))
  test_data = np.divide(test_data, np.amax(test_data, 0))
  ## transforming one-hot labels to class labels for loss function
  test_labels_class = np.argmax(test_labels, 1)

  
  X_test, Y_test = Variable(torch.Tensor(test_data)), Variable(torch.Tensor(test_labels_class))

  training_accuracies = []
  test_accuracies = []
  training_losses = []
  test_losses = []
  ## training loop
  while training_set.epochs_completed <= f['max_steps']:

    ## average accuracy calculation after epoch
    if lastEpochNum != training_set.epochs_completed:
      lastEpochNum = training_set.epochs_completed
      training_acc = epoch_acc/batchCounter
      tr_loss = epoch_loss/batchCounter
      training_losses.append(tr_loss)
      training_accuracies.append(training_acc)
      print("epoch " + str(lastEpochNum) + " avg accuracy on training data: "+ str(training_acc))
      batchCounter = 0
      epoch_acc = 0
      epoch_loss = 0

      ## also calculate accuracy on the test data for better visualization
      test_acc, test_loss = calculateTest(X_test, test_labels, 20, cnn, criterion, test_labels_onehot=Y_test)
      test_accuracies.append(test_acc)
      test_losses.append(test_loss)

    
    ## testing and printng after number of batches, given the parameter
    if batchCounter % f['eval_freq'] == 0:
      calculateTest(X_test, test_labels, 20, cnn, printing=True, lossCalc=False)
      test_output = cnn(X_test)
      test_out_np = test_output.cpu().detach().numpy()
      test_acc = accuracy(test_out_np, test_labels)
      print("-----------------------")
      print("test accuracy: " + str(test_acc))
      print("-----------------------")
    

    ## fetching batch and training
    batch_data, batch_labels = training_set.next_batch(batch_size)
    #batch_data_flat = np.reshape(batch_data, (batch_size, input_size))
    ### normalize
    batch_data = np.subtract(batch_data,np.mean(batch_data, 0))
    batch_data = np.divide(batch_data, np.amax(batch_data, 0))
    ## transforming one-hot labels to class labels for loss function
    batch_labels_class = np.argmax(batch_labels, 1)

    X, Y = Variable(torch.Tensor(batch_data)), Variable(torch.Tensor(batch_labels_class))
    if torch.cuda.is_available():
      X = X.cuda()
      Y = Y.cuda()
    
    optimizer.zero_grad()
    outputs = cnn(X)
    loss = criterion(outputs, Y.long())
    loss.backward()
    optimizer.step()
    outputs = outputs.cpu().detach().numpy()

    acc = accuracy(outputs, batch_labels)
    epoch_acc += acc
    epoch_loss += loss
    batchCounter += 1

    X.detach()
    Y.detach()
    loss.detach()

  drawPlot(training_accuracies, test_accuracies, './cnn-accuracies.png', 'ConvNet - accuracies on training and test data', 3)
  drawPlot(training_losses, test_losses, './cnn-loss_numpy.png', 'ConvNet - loss on training and test data', 4)

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()
