from matplotlib import pyplot as plt

import numpy as np

def drawPlot(trainAcc, savePath, title, figNum):
    plt.figure(figNum)
    plt.plot(trainAcc)

    plt.xlabel('step')
    plt.ylabel('accuracy')
    plt.title(title)
    plt.legend(['Training'])

    plt.savefig(savePath)
    return plt

def drawPlotMagn(trainAcc, savePath, title, figNum):
    plt.figure(figNum)
    plt.plot(trainAcc)

    plt.xlabel('time step')
    plt.ylabel('gradient magnitude')
    plt.title(title)
    plt.legend(['Training'])

    plt.savefig(savePath)
    return plt


def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzvf cifar-10-python.tar.gz
rm cifar-10-python.tar.gz 
Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODO:
  Implement accuracy computation.
  """
  dim = np.shape(targets)[0]
  max_indices = np.argmax(predictions, axis= 1)
  ### i was told it's ok looping here
  accuracy = sum(max_indices == targets) / dim

  return accuracy
