import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def insert_intercept(dataframe):
    dataframe.insert(1, 'intercept', 1)
    return dataframe

def split_data(dataframe):

    ## Fill In Your Code Here ##
    columns = list(dataframe.columns.values)
    X = np.array(dataframe[columns[1:]])
    y = np.array(dataframe[columns[0]])
    ############################
    assert type(X) == np.ndarray
    assert type(y) == np.ndarray

    return X, y

def CoordinateLasso(X, y, lambda_):
    np.random.seed(0)
    training_error_history = []
    z = np.square(X).sum(axis = 0)

    ## Fill In Your Code Here ##
    # initialize w
    w = np.random.normal(0, 1, X.shape[1])
    w_old = w.copy()
    iteration = 0
    while(True):
        for j in range(len(w)):
            if j < len(w) - 1:
                X_noj = np.concatenate((X[:, :j], X[:, j+1:]), axis = 1)
                w_noj = np.concatenate((w[:j], w[j+1:]))
            else:
                X_noj = X[:, :-1]
                w_noj = w[:-1]
            y_noj = X_noj.dot(w_noj)
            rho_j = X[:, j].T.dot(y - y_noj)
            if j == 0: # for the intercept case
                w[j] = rho_j / z[j]
            else:
                if rho_j < -(lambda_ / 2):
                    w[j] = (rho_j + lambda_/ 2) / z[j]
                elif rho_j > (lambda_ / 2):
                    w[j] = (rho_j - lambda_/ 2) / z[j]
                else: w[j] = 0

        y_predict = X.dot(w)
        average_rss = np.square(y - y_predict).sum()/X.shape[0]
        training_error_history.append(average_rss)
        if(max(abs(w - w_old)) < 0.000001): break
        else: w_old = w.copy()
    ############################
    return w, training_error_history

def plot_error_over_iterations(error_history):
    fontsize = 30
    markersize = 20
    marker = '*'
    fig = plt.figure(figsize =(30,7))

    ## Fill In Your Code Here ##
    y = error_history
    x = list(x for x in range(len(y)))
    plt.plot(x,y, marker = marker, markersize = markersize)
    plt.xlabel('Iterations', fontsize = fontsize)
    plt.ylabel('Squared error', fontsize = fontsize)
    plt.title('training error over iterations', fontsize = fontsize)
    ############################
    plt.show()
    return fig

def stack_weights_by_lambda(lambda_ , X, y):
    w_tot = []
    ## Fill In Your Code Here ##
    for lambda_value in lambda_:
        w, error_history = CoordinateLasso(X, y, lambda_value)
        w_tot.append(w)
    w_tot = np.array(w_tot)
    ############################
    assert w_tot.shape == (10,96)
    return w_tot


def plot_weights(lambda_, w_tot, dataframe, features):
    fontsize = 30
    markersize = 20
    marker = '*'
    fig = plt.figure(figsize =(30,7))
    fig, ax = plt.subplots(figsize = (30, 7))

    ## Fill In Your Code Here ##
    for feature in features:
        column_index = dataframe.columns.get_loc(feature) - 1
        x = np.log(lambda_)
        y = w_tot[:, column_index]
        ax.plot(x, y, label = feature, marker = marker, markersize = markersize)

    left, right = ax.get_xlim()
    ax.set_xlim(left = right, right = left)
    ax.set_xlabel('log(lambda)', fontsize = fontsize)
    ax.set_ylabel('weights', fontsize = fontsize)
    plt.title('Regularization paths', fontsize = fontsize)
    plt.legend(fontsize = 20)
    ############################
    plt.show()
    return fig


def plot_training_error(lambda_, w_tot,  X, y):
    fontsize = 30
    markersize = 20
    marker = '*'
    fig = plt.figure(figsize=(30,7))

    ## Fill In Your Code Here ##
    y_predict = X.dot(w_tot.T)
    rss = np.square(y_predict.T - y).sum(axis = 1)
    plt.plot(np.log(lambda_), rss, marker = marker, markersize = markersize)
    plt.xlabel('log(lambda)', fontsize = fontsize)
    plt.ylabel('Squared errors', fontsize = fontsize)
    plt.title('Training errors over log(lambda)', fontsize = fontsize)
    ############################
    plt.show()
    return fig


def plot_test_error(lambda_, w_tot,  X, y):
    fontsize = 30
    markersize = 20
    marker = '*'
    fig = plt.figure(figsize=(30,7))

    ## Fill In Your Code Here ##
    y_predict = X.dot(w_tot.T)
    rss = np.square(y_predict.T - y).sum(axis = 1)
    plt.plot(np.log(lambda_), rss, marker = marker, markersize = markersize)
    plt.xlabel('log(lambda)', fontsize = fontsize)
    plt.ylabel('Squared errors', fontsize = fontsize)
    plt.title('Training errors over log(lambda)', fontsize = fontsize)
    ############################
    plt.show()
    return fig

def plot_number_of_nonzero_index(lambda_, w_tot):
    fontsize = 30
    markersize = 20
    marker = '*'
    fig = plt.figure(figsize=(30,7))

    ## Fill In Your Code Here ##
    count = 0
    y = []
    for weights in w_tot:
        for weight in weights:
            if weight != 0: count += 1
        y.append(count)
        count = 0
    plt.plot(lambda_, y , marker = marker, markersize = markersize)
    plt.xlabel('lambda', fontsize = fontsize)
    plt.ylabel('Number of non-zero weights', fontsize = fontsize)
    plt.title('Number of non-zero weights', fontsize = fontsize)
    ############################
    plt.show()
    return fig
