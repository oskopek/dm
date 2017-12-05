from __future__ import division
import numpy as np
import random
from scipy.spatial.distance import cdist

# The number of preprocessed features in the input.
NUM_FEATURES = 250

# Set the random seed
random.seed(42)
np.random.seed(42)


# Calculate the squared distances, rows represent distances from centers,
# columns distances from points in xs.
def sqdist(centers, xs):
    return cdist(centers, xs, metric='sqeuclidian')


# Sample n_samples points from xs according to the distribution p,
# without replacement.
def sample_from_points(xs, p, n_samples):
    assert(n_samples <= xs.shape[0])
    return xs[np.random.choice(xs.shape[0], n_samples, replace=False, p=p)]


# Sample n_samples points from xs based on the inverse squared distance
# to centers.
def d2sample(xs, centers, n_clusters=200):
    distances = sqdist(centers, xs)
    min_distances = np.min(distances, axis=0)  # from points to centers
    probs = min_distances / np.sum(min_distances)
    return sample_from_points(xs, probs, n_clusters)


# Implements coreset construction by iterative sampling.
# Paper: Algorithm 1: TODO
def d3sample(xs, centers, n_clusters=200):
    n_points = xs.shape[0]
    alpha = 16 * (np.log2(n_clusters) + 2)
    distances = sqdist(centers, xs)
    min_distances = np.min(distances, axis=0)
    c_phi = np.sum(min_distances) / n_points
    B_is = closest_centroid_indices = np.argmin(distances, axis=0)
    assert closest_centroid_indices.shape[0] == n_points

    probs = np.zeros(n_points)
    B_i_counts = np.zeros(n_clusters)
    unique, counts = np.unique(B_is, return_counts=True)
    B_i_counts[unique] = counts
    B_i_lens = B_i_counts[B_is]
    sum_of_dists_in_cluster = np.zeros(n_clusters)
    sum_of_dists_in_cluster = np.vectorize(
        lambda i: np.sum(min_distances[B_is == i]))(range(n_clusters))

    probs = alpha * min_distances / c_phi + \
        2 * alpha * sum_of_dists_in_cluster[B_is] / (B_i_lens * c_phi) + \
        4 * n_points / B_i_lens

    probs = probs / np.sum(probs)
    return sample_from_points(xs, probs, n_clusters)


# Implements kmeans++.
def kmeans_pp(xs, n_clusters):
    # Sample a new point based on the inverse squared distance to centers,
    # iteratively; uses distances as a matrix that it fills in top to bottom.
    def kpp_sample_iter(xs, distances, iteration):
        min_distances = np.min(distances[:iteration], axis=0)
        probs = min_distances / np.sum(min_distances)
        new_point = sample_from_points(xs, probs, 1)
        distances[iteration, :] = sqdist(new_point, xs)
        return new_point

    # Initial distribution.
    n_points = xs.shape[0]
    probs = np.ones(n_points) / n_points

    centers = np.zeros((n_clusters, NUM_FEATURES))
    distances = np.zeros((n_clusters, n_points))
    centers[0] = sample_from_points(xs, probs, 1)
    distances[0] = sqdist([centers[0]], xs)
    for k in range(1, n_clusters):
        centers[k] = kpp_sample_iter(xs, distances, k)
    return centers


# Implements kmeans with all points and distances in memory.
def kmeans_mem(xs, initial_centers=None, n_clusters=200,
               n_iterations=50, early_stopping=0.2):
    # Run kmeans++ if initial_centers were not provided.
    if initial_centers is None:
        centers = kmeans_pp(xs, n_clusters)
        print('kmeans++:\tend')
    else:
        centers = initial_centers

    centers_old = np.zeros_like(centers)
    for iteration in range(n_iterations):
        # Early stopping based on the norm of the change in centers.
        diff = np.linalg.norm(centers - centers_old)
        if diff < early_stopping:
            break
        distances = sqdist(centers, xs)
        idxs = np.argmin(distances, axis=0)
        assert idxs.shape[0] == xs.shape[0]
        centers_old = np.copy(centers)
        for i in range(n_clusters):
            idxs_i = idxs == i
            if idxs_i.any():
                centers[i] = np.mean(xs[idxs_i], axis=0)

    return centers


# Mapper: runs d3sample twice on uniformly sampled centers and yields them.
def mapper(key, value):
    # key: None
    # value: Numpy array of the points.
    n_clusters = 3000
    n_points = value.shape[0]
    if (n_clusters < n_points):
        uniform_probs = np.ones(n_points) / n_points
        centers = sample_from_points(
            value, p=uniform_probs, n_samples=n_clusters)
        for i in range(2):
            centers = d3sample(value, centers, n_clusters=n_clusters)
    else:
        centers = value
    print('mapper:\tend')
    yield 0, centers


# Reducer: runs kmeans_mem on the centers provided (includes KMeans++).
def reducer(key, values):
    # key: key from mapper used to aggregate (constant)
    # values: list of cluster centers.
    batch_centers = np.reshape(values, (-1, NUM_FEATURES))
    centers = kmeans_mem(batch_centers)
    print('reducer:\tend')
    yield centers
