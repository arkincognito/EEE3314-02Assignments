import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm

class kmeans:
    def __init__(self, x1, x2, k):

        self.x1 = x1
        self.x2 = x2
        self.k = k
        self.X = np.array(list(zip(x1, x2)))

    # Euclidean distance
    def EuclideanDistance(self, a, b, ax = 1):

        distance = np.linalg.norm(a-b, axis = ax)
        return distance

    # return X, cluster labels, coordinates of cluster centers(shape = (15,2))
    def clustering(self, iterations = 50):
        self.C = np.array(list())
        self.cluster_indices = np.array(list())

        best_hetero = 10 ** 20

        # Each clustering will converge into local minimum.
        # Take n iterations of clustering with different starting points to get various local minimums and hopefully get the most optimal clustering.
        for iteration in range(iterations):
            cluster_labels, C = self.cluster_iteration(iteration = iteration)
            hetero_current = self.cluster_heterogeneity_iterating(cluster_labels, C)

            if hetero_current < best_hetero:
                best_hetero = hetero_current
                self.C = C.copy()
                self.cluster_labels = cluster_labels.copy()
            ############################

        assert self.cluster_labels.shape == (self.X.shape[0],)
        assert self.C.shape == (self.k,2)

        return self.X, self.cluster_labels, self.C

    def cluster_heterogeneity_iterating(self, cluster_labels, C):

        ## Fill In Your Code Here ##
        heterogeneity = 0
        cluster_indices = [0 for _ in range(self.k)]

        for label in range(self.k):
            cluster_indices[label] = np.where(cluster_labels == label)
            heterogeneity += np.sum(self.EuclideanDistance(self.X[cluster_indices[label]], C[label]))
        ############################

        return heterogeneity

    def cluster_heterogeneity(self):

        ## Fill In Your Code Here ##
        heterogeneity = 0
        cluster_indices = [0 for _ in range(self.k)]

        for label in range(self.k):
            cluster_indices[label] = np.where(self.cluster_labels == label)
            heterogeneity += np.sum(self.EuclideanDistance(self.X[cluster_indices[label]], self.C[label]))
        ############################

        return heterogeneity

    def cluster_iteration(self, iteration = 0):
        # initial cluster centers for each iteration and run the cluster.
        np.random.seed(iteration)
        C_x = np.random.randint(0, np.max(self.x1)-np.mean(self.x1), size=self.k)
        # y coordinates of random cluster center
        C_y = np.random.randint(0, np.max(self.x2)-np.mean(self.x2), size=self.k)
        C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
        old_C = C.copy()

        #Each cell in cluster_indices will store a list of indices. np.array doesn't allow assigning sequence into cell. so list it is.
        cluster_labels = np.array([0 for i in range(self.X.shape[0])])
        cluster_indices = [0 for i in range(self.k)]
        breakstate = False

        while(True):
            #first, need X and C to stack to make the (15, 3000, 2) shape(or (3000, 15, 2), doesn't really matter). Then, get the argmin out of axis = 0
            cluster_labels = np.argmin(self.EuclideanDistance(np.stack([self.X for _ in range(self.k)], axis = 0), np.stack([C for _ in range(self.X.shape[0])], axis = 1), ax = 2), axis = 0)

            for label in range(self.k):
                cluster_indices[label] = np.where(cluster_labels == label)
                # If the the label looses all the points, then the indices will have 0 items.
                # For lists, this will be represented by 0 boolean.
                if not cluster_indices[label][0].any():
                    # print(self.k, iteration, label)
                    # print(cluster_indices[label])
                    breakstate = True
                    break
                C[label] = np.mean(self.X[cluster_indices[label]], axis = 0)

            if breakstate: break

            if np.equal(old_C, C).all(): break
            else: old_C = C.copy()

        return cluster_labels, C

def plot_data(X, cluster_labels, C, k):
    colors = cm.rainbow(np.linspace(0, 1, k))
    fig = plt.figure(figsize=(10,5))
    dotsize = 5
    centersize = 400
    ## Fill In Your Code Here ##
    for label in range(k):
        indices = np.where(cluster_labels == label)
        x = X[indices, 0]
        y = X[indices, 1]
        plt.scatter(x, y, s = dotsize)
    c_x = C[:, 0]
    c_y = C[:, 1]
    plt.scatter(c_x, c_y, color = 'Black', marker = '*', s = centersize)
    plt.show()
    ############################

    return plt
def plot_heterogeneity(x1, x2):

    heterogeneity = []
    list_range = [i for i in range(1, 15)]

    for k in list_range:
        model = kmeans(x1, x2, k)
        X, cluster_labels, C = model.clustering()
        heterogeneity.append(model.cluster_heterogeneity())

    fig = plt.figure(figsize = (10,5))
    fontsize = 30
    markersize = 20
    k = list_range
    plt.scatter(k, heterogeneity, s = markersize)
    plt.xlabel('k', fontsize = fontsize)
    plt.ylabel('heterogeneity', fontsize = fontsize)
    plt.title('heterogeneity for k', fontsize = fontsize)
    plt.show()
    return heterogeneity

def find_elbow(heterogeneity):

    elbow_slope_change = 0
    elbow_k = 0

    for pos in range(len(heterogeneity) - 2):
        slope1 = heterogeneity[pos] - heterogeneity[pos + 1]
        slope2 = heterogeneity[pos + 1] - heterogeneity[pos + 2]
        sl_change = slope1 / slope2

        if elbow_slope_change < sl_change:
            elbow_slope_change = sl_change
            elbow_k = pos + 2

    return elbow_k
