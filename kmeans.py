import numpy as np

class KMeans:
    def __init__(self, clusters, init=None, tol=1e-6):
        self.clusters = clusters
        self.init = init
        self.centroids = None
        self.partition = None
        self.tol = tol

    def cluster(self, dataset):
        # Randomly choose initial set of centroids if undefined
        if not self.init:
            rows = np.arange(dataset.shape[0])
            self.init = np.array([dataset[i] for i in np.random.choice(rows, size=self.clusters, replace=False)], dtype=np.float)

        self.centroids = self.init

        # Vectorize stopping condition function
        stop_vfunc = np.vectorize(lambda c1, c2: np.sqrt(self.__distance(c1, c2)) <= self.tol)
        
        # Optimize
        while True:
            # Partition dataset
            partition = []
            for d in dataset:
                partition.append(np.argmin([self.__distance(c, d) for c in self.centroids]))
            self.partition = np.array(partition, np.float)

            # Update centroids
            centroids = []
            for i in range(self.clusters):
                vs = [d for j,d in zip(self.partition, dataset) if j == i]
                if vs:
                    centroids.append(np.mean(vs, axis=0))
                else:
                    centroids.append(self.centroids[i])
            prev_centroids = self.centroids
            self.centroids = np.array(centroids, np.float)

            # Check if converged
            if np.all(stop_vfunc(self.centroids, prev_centroids)):
                break

    def __distance(self, v1, v2):
        return np.sum(np.power(v1 - v2, 2))
