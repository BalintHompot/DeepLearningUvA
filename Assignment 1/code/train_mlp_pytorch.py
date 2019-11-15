"""
This module implements training and etestuation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.cuda
from accuracies import drawPlot, accuracy

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 150
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100
NEG_SLOPE_DEFAULT = 0.02

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

## accuracy moved to separate file

def train():
  """
  Performs training and etestuation of MLP model. 

  TODO:
  Implement training and etestuation of MLP model. Etestuate your model on the whole test set each etest_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  # Get negative slope parameter for LeakyReLU
  neg_slope = FLAGS.neg_slope
  
  cifar10 = cifar10_utils.get_cifar10("./cifar10/cifar-10-batches-py")
  training_set = cifar10['train']
  test_set = cifar10['test']
  f = vars(FLAGS)
  input_size = 3*32*32
  number_of_classes = 10
  batch_size = f['batch_size']
  ### definition of architecture:
  layers =  dnn_hidden_units
  mlp = MLP(input_size, layers, number_of_classes, neg_slope )
  if torch.cuda.is_available():
    mlp = mlp.cuda()
  lastEpochNum = 0
  batchCounter = 0
  epoch_acc = 0
  epoch_loss = 0

  optimizer = optim.SGD(mlp.parameters(), lr=f['learning_rate'])
  criterion = torch.nn.CrossEntropyLoss()

  ## preparing test data
  test_data, test_labels = test_set.images, test_set.labels
  test_data_flat = np.reshape(test_data, (np.shape(test_data)[0], input_size))
  ### normalize
  test_data_flat = np.subtract(test_data_flat,np.mean(test_data_flat, 0))
  test_data_flat = np.divide(test_data_flat, np.amax(test_data_flat, 0))
  ## transforming one-hot labels to class labels for loss function
  test_labels_class = np.argmax(test_labels, 1)

  X_test, Y_test = Variable(torch.Tensor(test_data_flat)), Variable(torch.Tensor(test_labels_class))
  if torch.cuda.is_available():
    X_test = X_test.cuda()
    Y_test = Y_test.cuda()

  training_accuracies = []
  test_accuracies = []
  training_losses = []
  test_losses = []
  ## training loop
  while training_set.epochs_completed <= f['max_steps']:

    ## printing after epoch
    if lastEpochNum != training_set.epochs_completed:
      lastEpochNum = training_set.epochs_completed
      train_acc = epoch_acc/batchCounter
      tr_loss = epoch_loss/batchCounter
      training_losses.append(tr_loss)
      training_accuracies.append(train_acc)
      print("epoch " + str(lastEpochNum) + " avg accuracy on training data: "+ str(train_acc))
      batchCounter = 0
      epoch_acc = 0
      epoch_loss = 0

      ## also calculate accuracy on the test data for better visualization
      test_output = mlp(X_test)
      test_out_np = test_output.cpu().detach().numpy()
      test_loss = criterion(test_output, Y_test.long())           
      test_acc = accuracy(test_out_np, test_labels)
      test_accuracies.append(test_acc)
      test_losses.append(test_loss)

    ## testing after number of batches
    if batchCounter % f['eval_freq'] == 0:
      test_output = mlp(X_test)
      test_out_np = test_output.cpu().detach().numpy()
      test_acc = accuracy(test_out_np, test_labels)
      print("-----------------------")
      print("test accuracy: " + str(test_acc))
      print("-----------------------")

    ## fetching batch and training
    batch_data, batch_labels = training_set.next_batch(batch_size)
    batch_data_flat = np.reshape(batch_data, (batch_size, input_size))
    ### normalize
    batch_data_flat = np.subtract(batch_data_flat,np.mean(batch_data_flat, 0))
    batch_data_flat = np.divide(batch_data_flat, np.amax(batch_data_flat, 0))
    ## transforming one-hot labels to class labels for loss function
    batch_labels_class = np.argmax(batch_labels, 1)

    X, Y = Variable(torch.Tensor(batch_data_flat)), Variable(torch.Tensor(batch_labels_class))
    if torch.cuda.is_available():
      X = X.cuda()
      Y = Y.cuda()
    
    optimizer.zero_grad()
    outputs = mlp(X)
    loss = criterion(outputs, Y.long())
    loss.backward()
    optimizer.step()
    outputs = outputs.cpu().detach().numpy()

    acc = accuracy(outputs, batch_labels)
    epoch_loss += loss
    epoch_acc += acc
    batchCounter += 1
  
  drawPlot(training_accuracies, test_accuracies, './mlp-accuracies-pytorch.png', 'MLP pytorch - accuracies on training and test data', 1)
  drawPlot(training_losses, test_losses, './mlp-loss-pytorch.png', 'MLP pytorch - loss on training and test data', 2)

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, testue in vars(FLAGS).items():
    print(key + ' : ' + str(testue))

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
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
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
  parser.add_argument('--neg_slope', type=float, default=NEG_SLOPE_DEFAULT,
                      help='Negative slope parameter for LeakyReLU')
  FLAGS, unparsed = parser.parse_known_args()

  main()