from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt

from kmeans import KMeans, OnlineKMeans, MiniBatchKMeans


def callback(output_dct, input_dct):
    # Unpack dictionary of statistics
    for key in input_dct:
        output_dct.setdefault(key, []).append(input_dct[key])

# Read in dataset from file
with open('faithful.txt', 'rt') as f:
    data = []
    for row in f:
        cols = row.strip('\r\n').split(' ')
        data.append(np.fromiter(map(lambda x: float(x), cols), dtype=np.float))
    data = np.array(data)

# Initialize dicts containing statistics
kmeans_dct = {}
online_dct = {}
minibatch_dct = {}

# Cluster using K-Means algorithm
k_means = KMeans(2, tol=1e-8, callback=lambda dct: callback(kmeans_dct, dct))
k_means.cluster(data)
kmeans_centroids = k_means.centroids

# Cluster using online K-Means algorithm
online_k_means = OnlineKMeans(2, callback=lambda dct: callback(online_dct, dct))
online_k_means.cluster(data[:10])
for rate,data_point in zip(np.linspace(1.0, 1e-6, len(data[10:])), data[10:]):
    online_k_means.update(data_point, rate)
online_centroids = online_k_means.centroids

# Cluster using mini-batch K-Means algorithm
mini_batch_k_means = MiniBatchKMeans(2, 10, iterations=10, callback=lambda dct: callback(minibatch_dct, dct))
mini_batch_k_means.cluster(data)
minibatch_centroids = mini_batch_k_means.centroids

# Plot data with centroids
plt.figure()
plt.scatter(np.transpose(data)[0], np.transpose(data)[1])
plt.scatter(np.transpose(kmeans_centroids)[0], np.transpose(kmeans_centroids)[1], color='r', marker='x')
plt.scatter(np.transpose(minibatch_centroids)[0], np.transpose(minibatch_centroids)[1], color='r', marker='+')
plt.scatter(np.transpose(online_centroids)[0], np.transpose(online_centroids)[1], color='r', marker='D')
plt.legend(['Dataset', 'K-Means', 'Mini-Batch', 'Online'], loc='upper left')
plt.savefig('scatter_plot.pdf')

# Plot cost function
plt.figure()
plt.plot(kmeans_dct['cost_func'])
plt.plot(online_dct['cost_func'], '-.')
plt.plot(minibatch_dct['cost_func'], '--')
plt.savefig('cost_func.pdf')
