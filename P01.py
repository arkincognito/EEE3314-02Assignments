import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

# P1
def read_csv_using_pandas(csv_path='exam_scores.csv'):

    data = pd.read_csv(csv_path)

    print(data.shape)
    print(data.head())

    return data


def parse_pd_data(data, fields=['Circuit',
                                'DataStructure',
                                'MachineIntelligence']):
    values = []

    ## Fill In Your Code Here ##
    values = np.array(list(data[field] for field in fields))
    ############################

    return values

def plot_data(values):
    assert len(values) == 3
    assert type(values[0]) == np.ndarray
    figsize = (6, 4)
    title_fontsize = 20
    label_fontsize = 15

    fig = plt.figure(figsize=figsize)
    ax = Axes3D(fig)

    ## Fill In Your Code Here ##
    x = values[0]
    y = values[1]
    z = values[2]
    # scatter
    ax.scatter(x,y,z)
    # set title. use title_fontsize above.
    plt.suptitle('Score Distributions', fontsize = title_fontsize)
    # set labels for each axes. use label_fontsize above.
    ax.set_xlabel('Circuit', fontsize = label_fontsize)
    ax.set_ylabel('DS', fontsize = label_fontsize)
    ax.set_zlabel('MI', fontsize = label_fontsize)
    ############################

    plt.show()
    return fig


# P2
def prepare_dataset_for_linear_regression(values):

    bias = np.ones(len(values[0]))
    X = np.array([bias, values[0], values[1]]).T
    y = np.array(values[2])

    return X, y


class LinearRegression:

    def __init__(self, lr=0.0001, iterations=100000):
        self.lr = lr
        self.iterations = iterations
        self.average_rss_history = []

    def fit(self, X, y):
        # N = number of training set
        N = len(y)

        ## Fill In Your Code Here ##
        # initialize w

        self.w = np.zeros(3)

        ############################

        for i in range(self.iterations):

            ## Fill In Your Code Here ##
            # implement gradient descent
            y_predict = self.predict(X)
            average_rss = ((y - y_predict)**2).sum()/N
            gradient_rss = -2 * X.T.dot(y - y_predict)/N
            self.w -= self.lr * gradient_rss
            ############################
            self.average_rss_history.append(average_rss)


    def predict(self, X):
        ## Fill In Your Code Here ##
        pred_y = X.dot(self.w)
        ############################
        return pred_y


def plot_average_rss_history(iterations, history):
    figsize = (6,4)
    title_fontsize = 20
    label_fontsize = 15

    # plot rss_avg history over iterations
    fig = plt.figure(figsize=figsize)
    plt.ylim(0,100)

    ## Fill In Your Code Here ##
    y = history
    x = list(i for i in range(iterations))

    # plot
    plt.plot(x,y)
    # set title
    plt.title('Average RSS History', fontsize = title_fontsize)
    # set labels for axes
    plt.xlabel('Iterations', fontsize = label_fontsize)
    plt.ylabel('Average_RSS', fontsize = label_fontsize)
    ############################


    plt.show()
    return fig


# P3
def plot_data_with_wireframe(values, w, wireframe_color='red'):
    assert len(w) == 3
    title_fontsize = 20
    label_fontsize = 15
    figsize = (6,4)

    def make_meshgrids(x, y, num=10):

        ## Fill In Your Code Here ##
        # make meshgrids for 3D plot.
        # HINT : use np.linspace function
        x_margin = (max(x) - min(x)) * 0.05
        y_margin = (max(y) - min(y)) * 0.05
        x_linspace = np.linspace(min(x) - x_margin, max(x) + x_margin, num)
        y_linspace = np.linspace(min(y) - y_margin, max(y) + y_margin, num)
        ############################
        x_grid, y_grid = np.meshgrid(x_linspace, y_linspace)

        return x_grid, y_grid

    x_grid, y_grid = make_meshgrids(values[0], values[1])
    # For one fig, one figure of plot (either in 2D plane or in 3D space)
    fig = plt.figure(figsize=figsize)
    # ax of axis
    ax = Axes3D(fig)

    ## Fill In Your Code Here ##
    # X, Y = np.meshgrid(values[0], values[1])
    # Z = np.array([1, X, Y]).dot(w)
    Z = np.array([1, x_grid, y_grid]).dot(w)
    # scatter
    # scatter takes arrays for inputs
    ax.scatter(values[0], values[1], values[2])
    # set title. use title_fontsize above.
    plt.suptitle('Score Distributions', fontsize = title_fontsize)
    # set labels for each axes. use label_fontsize above.
    ax.set_xlabel('Circuit', fontsize = label_fontsize)
    ax.set_ylabel('DS', fontsize = label_fontsize)
    ax.set_zlabel('MI', fontsize = label_fontsize)
    # plot wireframe
    # plot_wireframe takes (2 arrays and one 2D matrix)
    ax.plot_wireframe(x_grid, y_grid, Z, color = wireframe_color)
    ############################

    plt.show()
    return fig


def get_closed_form_solution(X, y):

    w = np.zeros(X.shape[1])

    ## Fill In Your Code Here ##
    w = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))


    ############################

    return w
