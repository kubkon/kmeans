from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt

from kmeans import KMeans

# Read in dataset from file
with open('faithful.txt', 'rt') as f:
    data = []
    for row in f:
        cols = row.strip('\r\n').split(' ')
        data.append(np.fromiter(map(lambda x: float(x), cols), dtype=np.float))
    data = np.array(data)

# Cluster using K-Means algorithm
k_means = KMeans(2)
k_means.fit(data)
centroids = k_means.centroids

# Plot
plt.figure()
plt.scatter(np.transpose(data)[0], np.transpose(data)[1])
plt.scatter(np.transpose(centroids)[0], np.transpose(centroids)[1], marker='x')
plt.savefig('scatter_plot.pdf')
