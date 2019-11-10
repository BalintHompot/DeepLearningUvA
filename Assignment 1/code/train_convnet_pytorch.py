"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODO:
  Implement accuracy computation.
  """

  max_indices = np.argmax(predictions, axis= 1)
  dim = len(max_indices)
  correct = 0
  for instance in range(dim):
    if targets[instance][max_indices[instance]] == 1:
      correct += 1
  accuracy = correct/dim

  return accuracy

  return accuracy

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
  optimizer = optim.Adam(cnn.parameters(), lr=f['learning_rate'])
  criterion = torch.nn.CrossEntropyLoss()

  ## preparing test data
  test_data, test_labels = test_set.images, test_set.labels
  ### normalize
  #test_data_flat = np.subtract(test_data_flat,np.mean(test_data_flat, 0))
  #test_data_flat = np.divide(test_data_flat, np.amax(test_data_flat))
  ## transforming one-hot labels to class labels for loss function
  test_labels_class = np.argmax(test_labels, 1)

  X_test, Y_test = Variable(torch.Tensor(test_data)), Variable(torch.Tensor(test_labels_class))
  '''
  if torch.cuda.is_available():
    X_test = X_test.cuda()
    Y_test = Y_test.cuda()
  '''
  ## training loop
  while training_set.epochs_completed <= f['max_steps']:

    ## printing after epoch
    if lastEpochNum != training_set.epochs_completed:
      lastEpochNum = training_set.epochs_completed
      print("epoch " + str(lastEpochNum) + " avg accuracy on training data: "+ str(epoch_acc/batchCounter))
      batchCounter = 0
      epoch_acc = 0

    '''
    ## testing after number of batches
    if batchCounter % f['eval_freq'] == 0:
      test_output = cnn(X_test)
      test_out_np = test_output.cpu().detach().numpy()
      test_acc = accuracy(test_out_np, test_labels)
      print("-----------------------")
      print("test accuracy: " + str(test_acc))
      print("-----------------------")
    '''
    
    ## fetching batch and training
    batch_data, batch_labels = training_set.next_batch(batch_size)
    #batch_data_flat = np.reshape(batch_data, (batch_size, input_size))
    ### normalize
    #batch_data_flat = np.subtract(batch_data_flat,np.mean(batch_data_flat, 0))
    #batch_data_flat = np.divide(batch_data_flat, np.amax(batch_data_flat))
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
    batchCounter += 1

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