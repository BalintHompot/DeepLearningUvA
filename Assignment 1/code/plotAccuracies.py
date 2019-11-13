from matplotlib import pyplot as plt
import numpy as np
def drawPlot(trainAcc, testAcc):

    plt.plot(trainAcc)
    plt.plot(testAcc)

    plt.xlabel('epocc')
    plt.ylabel('accuracy')
    plt.title('Accuracies on training and testing data')
    plt.legend(['Training (average of batches)', 'Testing'])
    plt.show()
    return plt