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
    labeled_centroids -- dictionary of centroids with assigned labels
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
        self.labeled_centroids = None
        self.tol = tol
        self.cost_func = 0
        self.__callback = callback

    def cluster(self, dataset):
        """
        Runs standard K-Means algorithm on the dataset.

        Arguments:
        dataset -- input dataset to be clustered
        """
        # Initialize algorithm
        self.initialize(dataset)
        rows = dataset.shape[0]
        
        # Optimize
        while True:
            # Partition dataset
            for i in np.arange(rows):
                for j in np.arange(self.clusters):
                    self.distances[j] = self.distance(self.centroids[j], dataset[i])
                self.partition[i] = np.argmin(self.distances)

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
            if np.all(self.stop_vfunc(self.centroids, prev_centroids)):
                break

    def predict(self, dataset):
        """
        Partitions dataset using pre-computed centroid values.

        Arguments:
        dataset -- input dataset to be clustered
        """
        # Initialize
        rows = dataset.shape[0]
        distances = np.empty(self.clusters, dtype=np.float)
        partition = []

        # Partition dataset
        for i in np.arange(rows):
            for j in np.arange(self.clusters):
                distances[j] = self.distance(self.centroids[j], dataset[i])
            nearest_index = np.argmin(distances)
            partition.append(self.labeled_centroids[nearest_index])

        return partition

    def label_centroids(self, f):
        """
        Labels centroids according to the specified function.

        Arguments:
        f -- function that takes an array of centroids as an argument, and
        returns a dictionary of labeled centroids where key corresponds to the
        index of a centroid within self.centroids array, and value is the label
        """
        self.labeled_centroids = f(self.centroids)

    def call_back(self):
        """
        Callback with the dictionary containing statistics: current centroids,
        partition, and value of the cost function.
        """
        if self.__callback:
            dct = {
                'centroids': self.centroids,
                'partition': self.partition,
                'cost_func': self.cost_func
                }
            self.__callback(dct)

    def distance(self, v1, v2):
        """
        Returns Euclidean distance squared between two NumPy arrays.

        Arguments:
        v1 -- 1st vector
        v2 -- 2nd vector
        """
        return np.sum(np.power(v1 - v2, 2))

    def initialize(self, dataset):
        """
        Initializes variables of the algorithm; such as, the initial
        set of centroids if undefined, etc.

        Arguments:
        dataset -- input dataset
        """
        # Get number of samples (rows) and features (cols)
        rows, cols = dataset.shape[0], dataset.shape[1]

        # Initialize partition and distances array
        self.partition = np.empty(rows, dtype=np.int)
        self.distances = np.empty(self.clusters, dtype=np.float)

        # Randomly choose initial set of centroids if undefined
        if not self.init:
            self.init = np.empty((self.clusters, cols), dtype=np.float)
            for i in np.arange(self.clusters):
                for j in np.random.choice(np.arange(rows), size=self.clusters, replace=False):
                    self.init[i] = dataset[j]
        self.centroids = self.init

        # Vectorize stopping condition function
        self.stop_vfunc = np.vectorize(lambda c1, c2: self.distance(c1, c2) <= self.tol)

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
        for j in np.arange(self.clusters):
            self.distances[j] = self.distance(self.centroids[j], data_point)

        nearest = np.argmin(self.distances)

        # Update the nearest centroid
        self.centroids[nearest] += learning_rate * (data_point - self.centroids[nearest])


class MiniBatchKMeans(KMeans):
    """
    This class implements a mini-batch version of the K-Means algorithm
    as described in the paper "Web-Scale K-Means Clustering" by D. Sculley.

    Attributes:
    clusters -- number of clusters
    mini_batch -- mini-batch size
    iterations -- maximum number of iterations allowed
    init -- NumPy array of initial centroids
    centroids -- NumPy array of centroids
    partition -- NumPy array of indices that partition the dataset
    tol -- tolerance for stopping condition
    """

    def __init__(self, clusters, mini_batch, iterations=None, init=None, tol=1e-6, callback=None):
        """
        Arguments:
        clusters -- number of clusters
        mini_batch -- mini-batch size

        Keyword arguments:
        iterations -- maximum number of iterations allowed
        init -- NumPy array of initial centroids (default: None)
        tol -- desired tolerance for stopping condition (default: 1e-6)
        callback -- callback function accepting a dictionary of parameters:
        centroids, partition, and cost_func (default: None)
        """
        super().__init__(clusters, init=init, tol=tol, callback=callback)
        self.mini_batch = mini_batch
        self.iterations = iterations

    def cluster(self, dataset):
        """
        Runs standard K-Means algorithm on the dataset.

        Arguments:
        dataset -- input dataset to be clustered
        """
        # Initialize algorithm
        self.initialize(dataset)
        rows, cols = dataset.shape[0], dataset.shape[1]
        batch = np.empty((self.mini_batch, cols), dtype=np.float)
        batch_distances = np.empty(self.clusters, dtype=np.float)
        batch_partition = np.empty(self.mini_batch, dtype=np.int)
        centroid_counts = np.zeros(self.clusters, dtype=np.int)
        it = 0

        # Optimize
        while it < self.iterations:

            # Randomly pick mini_batch samples from the sample space
            for i,j in zip(np.arange(self.mini_batch), np.random.choice(np.arange(rows), size=self.mini_batch, replace=False)):
                batch[i] = dataset[j]

            # Cache the center nearest to each sample in batch array
            for i in np.arange(self.mini_batch):
                for j in np.arange(self.clusters):
                    batch_distances[j] = self.distance(self.centroids[j], batch[i])
                batch_partition[i] = np.argmin(batch_distances)

            # Update centroids
            prev_centroids = np.copy(self.centroids)

            for i in np.arange(self.mini_batch):
                j = batch_partition[i]
                centroid = self.centroids[j]

                # Update per centroid counts
                centroid_counts[j] += 1

                # Get per centroid learning rate
                rate = 1 / centroid_counts[j]

                # Take gradient step
                self.centroids[j] = (1 - rate) * centroid + rate * batch[i]

            # Partition the entire dataset based on the derived centroids
            for i in np.arange(rows):
                for j in np.arange(self.clusters):
                    self.distances[j] = self.distance(self.centroids[j], dataset[i])
                self.partition[i] = np.argmin(self.distances)

            # Update cost function
            self.cost_func = 0
            for i,j in zip(np.arange(rows), self.partition):
                self.cost_func += self.distance(dataset[i], self.centroids[j])

            # Pass statistics to callback listener
            self.call_back()

            # Update iteration count
            it += 1

            # Check if converged
            if np.all(self.stop_vfunc(self.centroids, prev_centroids)):
                break
