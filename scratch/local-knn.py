# http://scikit-learn.org/stable/modules/neighbors.html
# https://github.com/aaalgo/kgraph/blob/master/python/test.py

from sklearn.neighbors import NearestNeighbors
import numpy as np
from numpy import random

N = 1000
Q = 20
D = 13
K = 10
TYPE = 'f'

dataset = random.rand(N, D).astype(TYPE)
query = random.rand(Q, D).astype(TYPE)

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

nbrs = NearestNeighbors(n_neighbors=K, algorithm='brute').fit(dataset)
distances, indices = nbrs.kneighbors(query)

indices
distances

# pip install pynndescent

from pynndescent import NNDescent
index = NNDescent(dataset, n_neighbors=K)
indices, distances = index.query(query)

indices
distances