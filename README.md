kmeans
======

My attempts at implementation of K-Means algorithm (and its variants).

Pull requests, comments, and suggestions are welcomed!

Installation
============
I'm assuming you are wise and you're using virtualenv and virtualenvwrapper. If not, go and [install them now](http://virtualenvwrapper.readthedocs.org/en/latest/).

In order to install the package, run in the terminal:

``` console
$ python setup.py sdist
```

Then:

``` console
$ pip install dist/KMeans-0.1.0.tar.gz
```

And you're done!

Basic usage
===========

A typical usage would look as follows:

``` python
from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt

from kmeans.algorithms import KMeans


# Read in dataset from file
with open('faithful.txt', 'rt') as f:
    data = []
    for row in f:
        cols = row.strip('\r\n').split(' ')
        data.append(np.fromiter(map(lambda x: float(x), cols), dtype=np.float))
    data = np.array(data)

# Cluster using K-Means algorithm
k_means = KMeans(2, tol=1e-8)
k_means.fit(data)

# Print the results
print("Computed centroids:")
pprint(k_means.centroids)

```

More examples
=============
You can find more examples in the ```examples/``` folder.

License
=======

License information can be found in License.txt.