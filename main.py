from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt

from kmeans import KMeans, OnlineKMeans


# Read in dataset from file
with open('faithful.txt', 'rt') as f:
    data = []
    for row in f:
        cols = row.strip('\r\n').split(' ')
        data.append(np.fromiter(map(lambda x: float(x), cols), dtype=np.float))
    data = np.array(data)

out_dct = {}
def callback(dct):
    # Unpack dictionary of statistics
    for key in dct:
        out_dct.setdefault(key, []).append(dct[key])

# Cluster using K-Means algorithm
k_means = KMeans(2, tol=1e-8, callback=callback)
k_means.cluster(data)
centroids = k_means.centroids
pprint(centroids)

# Cluster using online K-Means algorithm
online_k_means = OnlineKMeans(2)
online_k_means.cluster(data[:10])
for rate,data_point in zip(np.linspace(1.0, 1e-6, len(data[10:])), data[10:]):
    online_k_means.update(data_point, rate)
centroids = online_k_means.centroids
pprint(centroids)

# Plot data with centroids
plt.figure()
plt.scatter(np.transpose(data)[0], np.transpose(data)[1])
plt.scatter(np.transpose(centroids)[0], np.transpose(centroids)[1], color='r', marker='x')
plt.savefig('scatter_plot.pdf')

# Plot cost function
plt.figure()
plt.plot(out_dct['cost_func'])
plt.savefig('cost_func.pdf')
