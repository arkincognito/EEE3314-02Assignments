import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_training_samples(X_train, y_train):
    assert X_train.shape == (1000, 785)
    assert y_train.shape == (1000, )

    fig = plt.figure(figsize=(100, 40))
    f, axes = plt.subplots(5,6, sharex = True, sharey = True)
    f.set_size_inches((16, 20))
    ## Fill In Your Code Here ##

    for i in range(5): # of rows
        for j in range(6):
            axes[i][j].imshow(X_train[i * 6 + j][:784].reshape(28, 28))
            axes[i][j].set_title(f'{i * 6 + j}th, {y_train[i * 6 + j]}')

    ############################
    plt.show()
    return fig

def sgn(x):
    return (x >= 0)*2-1

# it is to predict accuracy of X data.
def predict(X, y, w):

    ## Fill In Your Code Here ##
    pred_y = sgn(X.dot(w))
    accuracy = (y == pred_y).sum()
    ############################

    return accuracy

# return w, number_of_misclassifications, test_accuracy
def perceptron(X, y, w, epoch):

    number_of_misclassifications = []
    test_accuracy = []

    ## Fill In Your Code Here ##
    lr = 0.01
    X_train = X['train']
    X_test = X['test']
    y_train = y['train']
    y_test = y['test']
    for _ in range(epoch):
        for i in range(len(y_train)):
            y_pred = sgn(X_train[i].dot(w))
            w -= lr * (X_train[i].T.dot(y_pred - y_train[i]))
        number_of_misclassifications.append(y_train.shape[0] - predict(X_train, y_train, w))
        test_accuracy.append(predict(X_test, y_test, w) / y_test.shape[0])

    ############################
    return w, number_of_misclassifications, test_accuracy

def perceptron_wholedata(X, y, w, epoch):

    number_of_misclassifications = []
    test_accuracy = []

    ## Fill In Your Code Here ##
    lr = 0.01
    X_train = X['train']
    X_test = X['test']
    y_train = y['train']
    y_test = y['test']
    for _ in range(epoch):
        y_pred = sgn(X_train.dot(w))
        w -= lr * (X_train.T.dot(y_pred - y_train))
        number_of_misclassifications.append(y_train.shape[0] - predict(X_train, y_train, w))
        test_accuracy.append(predict(X_test, y_test, w) / y_test.shape[0])

    ############################
    return w, number_of_misclassifications, test_accuracy

# plot number_of_misclassifications returned by perceptron
def plot_number_of_misclassifications_over_epochs(errors):

    fig = plt.figure(figsize=(17,5))

    ## Fill In Your Code Here ##

    x = [i for i in range(len(errors))]
    plt.plot(x, errors)
    ############################
    plt.show()
    return fig

# plot test_accuracy returned by perceptron
def plot_accuracy_over_epochs(test_accuracy):

    fig = plt.figure(figsize=(17,5))

    ## Fill In Your Code Here ##
    x = [i for i in range(len(test_accuracy))]
    plt.plot(x, test_accuracy)
    ############################
    plt.show()

    return fig
