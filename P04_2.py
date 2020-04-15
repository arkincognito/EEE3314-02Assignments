import numpy as np
import collections  # it is optional to use collections
from operator import itemgetter, attrgetter
# prediction function is to predict label of one sample using k-NN

def predict(X_train, y_train, one_sample, k, lambda_value = 1):
    one_sample = np.array(one_sample)
    X_train = np.array(X_train)
    y_distance = []
    for vector in X_train:
        y_distance.append(distancesquare(vector, one_sample))
    index = 0
    y_distance = np.array(y_distance)
    # np.argsort returns the index of sorted ndarray
    index = np.argsort(y_distance)
    # by plugging in the index, we sort the distance and the labels by the distance.
    y = np.array(y_train[index])
    y_distance = y_distance[index]
    label = [0 for y in range(10)]
    # Simple KNN gives wrong answer. Use Weighted KNN instead
    # Normalize the distances of 1~kth NN by dividing the distances with k+1th NN's distance
    # https://epub.ub.uni-muenchen.de/1769/1/paper_399.pdf page 7
    for i in range(k):
        label[y[i]] += weight(y_distance[i]/y_distance[k], lambda_value)
    prediction = label.index(max(label))
    ############################
    return prediction

def distancesquare(pos1, pos2):
    d = np.sum(np.square(pos1 - pos2))
    return d

def weight(distance, lambda_value):
    w = np.exp(- distance / lambda_value)
    return w

# accuracy function is to return average accuracy for test or validation sets
def accuracy(X_train, y_train, X_test, y_test, k, lambda_value = 1):  # You can use def prediction above.

    ## Fill In Your Code Here ##
    acc = 0
    for test_x, test_y in zip(X_test, y_test):
        acc += test_y == predict(X_train, y_train, test_x, k, lambda_value)
    acc = acc / len(y_test)
    ############################
    return acc

# stack_accuracy_over_k is to stack accuracy over k. You can use def accuracy above.
def stack_accuracy_over_k(X_train, y_train, X_val, y_val):
    accuracies = []

    ## Fill In Your Code Here ##
    for k in range(1, 21):
        accuracies.append(accuracy(X_train, y_train, X_val, y_val, k))
    ############################
    assert len(accuracies) == 20
    return accuracies

def stack_accuracy_over_lambda(X_train, y_train, X_val, y_val):
    accuracies = []
    k = 3
    lambdas = list(100/(2**i) for i in range(20))
    ## Fill In Your Code Here ##
    for lambda_value in lambdas:
        accuracies.append(accuracy(X_train, y_train, X_val, y_val, k, lambda_value = lambda_value))
    ############################
    assert len(accuracies) == 20
    return accuracies

def stack_accuracy_on_k_and_lambda(X_train, y_train, X_val, y_val):
    accuracies = []
    lambdas = list(100/(2**i) for i in range(10))
    ## Fill In Your Code Here ##
    for k in range(1, 21):
        for lambda_value in lambdas:
            accuracies.append(accuracy(X_train, y_train, X_val, y_val, k, lambda_value = lambda_value))
    accuracies = np.array(accuracies)
    accuracies = accuracies.reshape((20,10))
    ############################
    assert accuracies.shape == (20, 10)
    return accuracies
