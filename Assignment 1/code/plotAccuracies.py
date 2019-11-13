from matplotlib import pyplot as plt
import os

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