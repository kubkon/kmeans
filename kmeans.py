import numpy as np

class KMeans(object):
    def __init__(self, clusters, init=None):
        self.clusters = clusters
        self.init = init
        self.centroids = np.array([])
        self.partition = np.array([])

    def fit(self, features):
        # Randomly choose initial set of centroids if undefined
        if not self.init:
            cols = features.shape[1]
            self.init = np.random.rand(self.clusters, cols)

        self.centroids = self.init
        
        # Optimize
        for n in range(10):
            # Partition dataset
            partition = []
            for d in features:
                partition.append(np.argmin([self.__distance(c, d) for c in self.centroids]))
            self.partition = np.array(partition, np.float)

            # Update centroids
            centroids = []
            for i in range(self.clusters):
                vs = [d for j,d in zip(self.partition, features) if j == i]
                if vs:
                    centroids.append(np.mean(vs, axis=0))
                else:
                    centroids.append(self.centroids[i])
            self.centroids = np.array(centroids, np.float)

    def predict(self, features):
        pass

    def __distance(self, v1, v2):
        return np.sum(np.power(v1 - v2, 2))
