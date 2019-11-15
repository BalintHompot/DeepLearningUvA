"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils
from accuracies import drawPlot, accuracy

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100
NEG_SLOPE_DEFAULT = 0.02

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None


def train():
  """
  Performs training and evaluation of MLP model. 

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
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
  layers = dnn_hidden_units + [number_of_classes]
  mlp = MLP(input_size, layers, number_of_classes, neg_slope, f['learning_rate'])
  lastEpochNum = 0
  batchCounter = 0
  epoch_acc = 0
  epoch_loss = 0

  ## preparing test data
  test_data, test_labels = test_set.images, test_set.labels
  test_data = np.reshape(test_data, (np.shape(test_data)[0], input_size))
  ### normalize
  test_data = np.subtract(test_data,np.mean(test_data, 0))
  test_data = np.divide(test_data, np.amax(test_data, 0))

  training_accuracies = []
  test_accuracies = []

  training_losses = []
  test_losses = []

  num_data = np.shape(training_set.images)[0]
  max_epochs = f['max_steps']/(num_data/batch_size)


  while training_set.epochs_completed <= max_epochs:
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
      test_output = mlp.forward(test_data)
      test_loss = mlp.loss.forward(test_output, test_labels)
      test_acc = accuracy(test_output, test_labels)
      test_accuracies.append(test_acc)
      test_losses.append(test_loss)

    ## testing after number of batches, given the parameter
    if batchCounter % f['eval_freq'] == 0:
      test_output = mlp.forward(test_data)
      test_acc = accuracy(test_output, test_labels)
      print("-----------------------")
      print("test accuracy: " + str(test_acc))
      print("-----------------------")

    batch_data, batch_labels = training_set.next_batch(batch_size)
    batch_data_flat = np.reshape(batch_data, (batch_size, input_size))
    ### normalize
    batch_data_flat = np.subtract(batch_data_flat,np.mean(batch_data_flat, 0))
    batch_data_flat = np.divide(batch_data_flat, np.amax(batch_data_flat, 0))

    ### forward pass
    output = mlp.forward(batch_data_flat)
    loss = mlp.loss.forward(output, batch_labels)
    ## backward
    loss_gradient = mlp.loss.backward(output, batch_labels)
    mlp.backward(loss_gradient)

    acc = accuracy(output, batch_labels)
    epoch_acc += acc
    epoch_loss += loss
    batchCounter += 1

  drawPlot(training_accuracies, test_accuracies, './mlp-accuracies_numpy.png', 'MLP numpy - accuracies on training and test data', 5)
  drawPlot(training_losses, test_losses, './mlp-loss_numpy.png', 'MLP numpy - loss on training and test data', 6)




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