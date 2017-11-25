import numpy as np
import random
from scipy.spatial.distance import cdist

# TODO: Do weighted mean of the clusters

#random.seed(42)
#np.random.seed(42)

def dist(xs, ys):
    return np.square(xs-ys).sum()

def kmeans_mem(xs, n_clusters=200, n_iterations=50):
    indices = np.random.random_integers(0, xs.shape[0], n_clusters)
    centroids = xs[indices]
    centroids_old = np.zeros_like(centroids)
    for iteration in range(n_iterations):
        diff = np.linalg.norm(centroids-centroids_old)
        if diff < 0.2:
            break
        print(iteration, diff)
        clusters = [[] for i in range(n_clusters)]
        distances = cdist(centroids, xs)
        idxs = np.argmin(distances, axis=0)
        centroids_old = np.copy(centroids)
        for i in range(n_clusters):
            idxs_i = idxs == i
            if idxs_i.any():
                centroids[i] = np.mean(xs[idxs_i], axis=0)
            else:
                print('Centroid', i, 'has no points.')
                mean_dist_to_centroids = np.mean(distances, axis=0)
                new_center = np.argmax(mean_dist_to_centroids)
                print('Choosing point', new_center, 'distance:', mean_dist_to_centroids[new_center])
                centroids[i] = xs[new_center]

    return centroids

def mapper(key, value):
    # key: None
    # value: Numpy array of the points.
    centers = kmeans_mem(value, n_clusters=400)
    yield 0, 


def reducer(key, values):
    # key: key from mapper used to aggregate (constant)
    # values: list of cluster centers.
    #yield np.means
    centers = kmeans_mem(np.asarray(values))
    yield centers

