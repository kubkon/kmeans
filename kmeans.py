import numpy as np

class KMeans:
    def __init__(self, clusters, init=None, tol=1e-6, callback=None):
        self.clusters = clusters
        self.init = init
        self.centroids = None
        self.partition = None
        self.tol = tol
        self.__callback = callback

    def cluster(self, dataset):
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
        stop_vfunc = np.vectorize(lambda c1, c2: np.sqrt(self.__distance(c1, c2)) <= self.tol)
        
        # Optimize
        while True:
            # Partition dataset
            for i in np.arange(rows):
                for j in np.arange(self.clusters):
                    distances[j] = self.__distance(self.centroids[j], dataset[i])
                self.partition[i] = np.argmin(distances)

            # Update centroids
            prev_centroids = np.copy(self.centroids)

            for i in np.arange(self.clusters):
                vs = [d for j,d in zip(self.partition, dataset) if j == i]
                self.centroids[i] = np.mean(vs, axis=0)

            # Pass statistics to callback listener
            if self.__callback:
                cost_func = 0
                for i,j in zip(np.arange(rows), self.partition):
                    cost_func += self.__distance(dataset[i], self.centroids[j])

                dct = {
                    'centroids': self.centroids,
                    'partition': self.partition,
                    'cost_func': cost_func
                    }
                self.__callback(dct)

            # Check if converged
            if np.all(stop_vfunc(self.centroids, prev_centroids)):
                break

    def __distance(self, v1, v2):
        return np.sum(np.power(v1 - v2, 2))
