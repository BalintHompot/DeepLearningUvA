from matplotlib import pyplot as plt
import os
import numpy as np

def drawPlot(trainAcc, testAcc, savePath):

    plt.plot(trainAcc)
    plt.plot(testAcc)

    plt.xlabel('epocc')
    plt.ylabel('accuracy')
    plt.title('Accuracies on training and testing data')
    plt.legend(['Training (average of batches)', 'Testing'])
    if os.path.isfile(savePath):
        os.remove(savePath)
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

  max_indices = np.argmax(predictions, axis= 1)
  dim = len(max_indices)
  correct = 0
  for instance in range(dim):
    if targets[instance][max_indices[instance]] == 1:
      correct += 1
  accuracy = correct/dim

  return accuracy

def printAndStoreAcc(train_acc, test_acc, train_list, test_list, classifier):
    pass

def printTestAcc(test_data, test_labels, classifier):
    pass