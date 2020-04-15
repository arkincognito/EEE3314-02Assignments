import numpy as np
import matplotlib.pyplot as plt


# Helper function to plot a decision boundary.
def plot_decision_boundary(pred_func, train_data, color):
    # Set min and max values and give it some padding
    x_min, x_max = train_data[:, 0].min() - .5, train_data[:, 0].max() + .5
    y_min, y_max = train_data[:, 1].min() - .5, train_data[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlGn)
    plt.scatter(train_data[:, 0], train_data[:, 1], c=color, cmap=plt.cm.RdYlGn)

def ReLU(x):
    x = x * (x > 0)
    return x

def sigmoid(x):
    x = 1 / (1 + np.exp(-x))
    return x

def softmax(x):
    x = np.exp(x) / np.exp(x).sum(axis = 1, keepdims = True)
    return x

# Helper function for forward propagation
def forward_propagation(model, X):

    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']

    h1 = X.dot(W1) + b1
    z1 = ReLU(h1)
    h2 = z1.dot(W2) + b2
    z2 = sigmoid(h2)
    h3 = z2.dot(W3) + b3
    y_hat = softmax(h3)

    cache = {'h1': h1, 'z1': z1, 'h2': h2, 'z2': z2, 'h3': h3, 'y_hat': y_hat}

    return y_hat, cache

def onehotencoding(y):
    y = y.reshape((-1, 1))
    i = np.hstack((1 * (y == 0), 1 * (y == 1)))
    return i

# Helper function to evaluate the total loss on the dataset
def compute_loss(model, X, y):
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
    y_hat, _ = forward_propagation(model, X)
    total_loss = - np.sum(y * np.log(y_hat))
    return total_loss

# Helper function to predict an output (0 or 1)
def predict(model, X):
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
    y_hat, _ = forward_propagation(model, X)
    prediction = np.argmax(y_hat, axis = 1)
    return prediction

def dsigmoid(y):
    return (y * (1 - y))

def dReLU(y):
    return (1 * (y > 0))


def back_propagation(model, cache, X, y):
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
    h1, z1, h2, z2, h3, y_hat = cache['h1'], cache['z1'], cache['h2'], cache['z2'], cache['h3'], cache['y_hat']
    ## Fill In Your Code Here ##
    dh3 = (y_hat - y)
    db3 = dh3.sum(axis = 0, keepdims= True)
    dW3 = z2.T.dot(dh3)
    dh2 = dh3.dot(W3.T) * dsigmoid(z2)
    db2 = dh2.sum(axis = 0, keepdims= True)
    dW2 = z1.T.dot(dh2)
    dh1 = dh2.dot(W2.T) * dReLU(z1)
    db1 = dh1.sum(axis = 0, keepdims= True)
    dW1 = X.T.dot(dh1)

    # print('db3', db3.shape, db3)
    # print('dW3', dW3.shape, dW3[:3])
    # print('dh2', dh2.shape, dh2[:3])
    # print('db2', db2.shape, db2)
    # print('dW2', dW2.shape, dW2[:3])
    # print('dh1', dh1.shape, dh1[:3])
    # print('db1', db1.shape, db1)
    # print('dW1', dW1.shape, dW1)
    ############################

    gradients = dict()
    gradients['dW3'] = dW3
    gradients['db3'] = db3
    gradients['dW2'] = dW2
    gradients['db2'] = db2
    gradients['dW1'] = dW1
    gradients['db1'] = db1
    return gradients


def randn_initialization(nn_input_dim, nn_hdim1, nn_hdim2, nn_output_dim):
    W1 = np.random.randn(nn_input_dim, nn_hdim1)
    b1 = np.zeros((1, nn_hdim1))
    W2 = np.random.randn(nn_hdim1, nn_hdim2)
    b2 = np.zeros((1, nn_hdim2))
    W3 = np.random.randn(nn_hdim2, nn_output_dim)
    b3 = np.zeros((1, nn_output_dim))

    return W1, b1, W2, b2, W3, b3


def const_initialization(nn_input_dim, nn_hdim1, nn_hdim2, nn_output_dim):
    # Constant initialization. why problematic?
    W1 = np.ones((nn_input_dim, nn_hdim1))
    b1 = np.zeros((1, nn_hdim1))
    W2 = np.ones((nn_hdim1, nn_hdim2))
    b2 = np.zeros((1, nn_hdim2))
    W3 = np.ones((nn_hdim2, nn_output_dim))
    b3 = np.zeros((1, nn_output_dim))

    return W1, b1, W2, b2, W3, b3


def build_model(X, y, nn_input_dim, nn_hdim1, nn_hdim2, nn_output_dim,
                lr=0.001, epoch=50000, print_loss=False, init_type='randn'):

    # Initialization
    np.random.seed(0)
    if init_type == 'randn':
        W1, b1, W2, b2, W3, b3 = randn_initialization(nn_input_dim, nn_hdim1, nn_hdim2, nn_output_dim)
    elif init_type == 'const':
        W1, b1, W2, b2, W3, b3 = const_initialization(nn_input_dim, nn_hdim1, nn_hdim2, nn_output_dim)

    model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}
    training_loss = []

    # cache = dict()
    # Full batch gradient descent.
    for i in range(epoch):

        # Forward propagation
        y_hat, cache = forward_propagation(model, X)

        # Backpropagation
        gradients = back_propagation(model, cache, X, y)

        # Parameter update
        W1 -= lr * gradients['dW1']
        b1 -= lr * gradients['db1']
        W2 -= lr * gradients['dW2']
        b2 -= lr * gradients['db2']
        W3 -= lr * gradients['dW3']
        b3 -= lr * gradients['db3']

        # Assign new parameters
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}

        # Print the loss.
        if print_loss and (i+1) % 1000 == 0:
            loss = compute_loss(model, X, y)
            print("Loss (iteration %i): %f" %(i+1, loss))
            training_loss.append(loss)

    return model, training_loss
