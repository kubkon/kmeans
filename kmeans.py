import numpy as np

class KMeans:
    """
    This class implements a standard version of the K-Means algorithm
    as described in the book "Pattern Recognition and Machine Learning"
    by Christopher M. Bishop.

    Attributes:
    clusters -- number of clusters
    init -- NumPy array of initial centroids
    centroids -- NumPy array of centroids
    partition -- NumPy array of indices that partition the dataset
    tol -- tolerance for stopping condition
    cost_func -- objective cost function
    """

    def __init__(self, clusters, init=None, tol=1e-6, callback=None):
        """
        Arguments:
        clusters -- number of clusters

        Keyword arguments:
        init -- NumPy array of initial centroids (default: None)
        tol -- desired tolerance for stopping condition (default: 1e-6)
        callback -- callback function accepting a dictionary of parameters:
        centroids, partition, and cost_func (default: None)
        """
        self.clusters = clusters
        self.init = init
        self.centroids = None
        self.partition = None
        self.tol = tol
        self.cost_func = 0
        self.__callback = callback

    def cluster(self, dataset):
        """
        Runs standard K-Means algorithm on the dataset.

        Arguments:
        dataset -- input dataset to be clustered
        """
        # Initialize numpy arrays
        rows, cols = dataset.shape[0], dataset.shape[1]
        self.partition = np.empty(rows, dtype=np.int)
        distances = np.empty(self.clusters, dtype=np.float)

        # Randomly choose initial set of centroids if undefined
        if not self.init:
            self.init = np.empty((self.clusters, cols), dtype=np.float)
            for i in np.arange(self.clusters):
                for j in np.random.choice(np.arange(rows), size=self.clusters, replace=False):
                    self.init[i] = dataset[j]
        self.centroids = self.init

        # Vectorize stopping condition function
        stop_vfunc = np.vectorize(lambda c1, c2: np.sqrt(self.distance(c1, c2)) <= self.tol)
        
        # Optimize
        while True:
            # Partition dataset
            for i in np.arange(rows):
                for j in np.arange(self.clusters):
                    distances[j] = self.distance(self.centroids[j], dataset[i])
                self.partition[i] = np.argmin(distances)

            # Update centroids
            prev_centroids = np.copy(self.centroids)

            for i in np.arange(self.clusters):
                vs = [d for j,d in zip(self.partition, dataset) if j == i]
                self.centroids[i] = np.mean(vs, axis=0)

            # Update cost function
            self.cost_func = 0
            for i,j in zip(np.arange(rows), self.partition):
                self.cost_func += self.distance(dataset[i], self.centroids[j])

            # Pass statistics to callback listener
            self.call_back()

            # Check if converged
            if np.all(stop_vfunc(self.centroids, prev_centroids)):
                break

    def distance(self, v1, v2):
        """
        Returns Euclidean distance squared between two NumPy arrays.

        Arguments:
        v1 -- 1st vector
        v2 -- 2nd vector
        """
        return np.sum(np.power(v1 - v2, 2))

    def call_back(self):
        """
        """
        if self.__callback:
            dct = {
                'centroids': self.centroids,
                'partition': self.partition,
                'cost_func': self.cost_func
                }
            self.__callback(dct)


class OnlineKMeans(KMeans):
    """
    This class implements an online (or sequential) version of the K-Means
    algorithm as described in the book "Pattern Recognition and Pattern
    Matching" by Christopher M. Bishop.

    Attributes:
    clusters -- number of clusters
    init -- NumPy array of initial centroids
    centroids -- NumPy array of centroids
    partition -- NumPy array of indices that partition the dataset
    tol -- tolerance for stopping condition
    """

    def update(self, data_point, learning_rate):
        """
        Updates the nearest centroid.

        Arguments:
        data_point -- next available data point
        learning_rate -- learning rate parameter
        """
        # Find the nearest centroid to the data point
        distances = np.empty(self.clusters, dtype=np.float)

        for j in np.arange(self.clusters):
            distances[j] = self.distance(self.centroids[j], data_point)

        nearest = np.argmin(distances)

        # Update the nearest centroid
        self.centroids[nearest] += learning_rate * (data_point - self.centroids[nearest])
