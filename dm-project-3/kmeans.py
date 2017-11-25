from __future__ import division
import numpy as np
import random
from scipy.spatial.distance import cdist

NUM_FEATURES = 250
# TODO: Do weighted mean of the clusters

random.seed(42)
np.random.seed(42)

def sample_from_points(xs, p, n_samples):
    assert(n_samples <= xs.shape[0])
    return xs[np.random.choice(range(0, xs.shape[0]), n_samples, replace=False, p=p)]

def dist(xs, ys):
    return np.square(xs-ys).sum()

def d2sample(xs, centroids, n_clusters=200):
    distances = cdist(centroids, xs, metric='sqeuclidean')
    min_distances = np.min(distances, axis=0)
    probs = min_distances / np.sum(min_distances)
    return sample_from_points(xs, probs, n_clusters)

def kpp_sample(xs, distances, new_centroid):
    new_distances = cdist([new_centroid], xs, metric='sqeuclidean')
    distances = np.concatenate((distances, new_distances), axis=0)
    min_distances = np.min(distances, axis=0)
    probs = min_distances / np.sum(min_distances)
    return distances, sample_from_points(xs, probs, 1)

def d3sample(xs, centroids, n_clusters=200):
    n_points = xs.shape[0]
    #alpha = np.log2(n_clusters)+1
    alpha = 16 * (np.log2(n_clusters) + 2)
    distances = cdist(centroids, xs, metric='sqeuclidean')
    min_distances = np.min(distances, axis=0)
    c_phi = np.sum(min_distances) / n_points
    closest_centroid_indices = np.argmin(distances, axis=0)
    assert closest_centroid_indices.shape[0] == n_points
    probs = np.zeros(n_points)
    for i in range(n_points):
        B_i = closest_centroid_indices[i]
        idx_Bi = closest_centroid_indices == B_i
        Bi_len = np.sum(idx_Bi)
        first_term = alpha * min_distances[i]/c_phi
        second_term = 2 * alpha * np.sum(min_distances[idx_Bi]) / (Bi_len * c_phi)
        third_term = 4 * n_points/Bi_len
        probs[i] = first_term + second_term + third_term

    probs = probs / np.sum(probs)
    return sample_from_points(xs, probs, n_clusters)

def kmeans_pp(xs, n_clusters):
    n_points = xs.shape[0]
    probs = np.ones(n_points)/n_points
    centroids = sample_from_points(xs, probs, 1)
    distances = np.empty((0, n_points), dtype=np.float32)
    for k in range(n_clusters-1):
        distances, new_centroid = kpp_sample(xs, distances, centroids[-1])
        centroids = np.concatenate((centroids, new_centroid), axis=0)
    return centroids

def kmeans_mem(xs, initial_centroids=None, n_clusters=200, n_iterations=50, early_stopping=0.2):
    # TODO: RUN with restarts
    n_points = xs.shape[0]
    if initial_centroids is None:
        centroids = kmeans_pp(xs, n_clusters)
        print('kpp')
    else:
        centroids = initial_centroids
    centroids_old = np.zeros_like(centroids)
    for iteration in range(n_iterations):
        diff = np.linalg.norm(centroids-centroids_old)
        if diff < early_stopping:
            break
        # print(iteration, diff)
        clusters = [[] for i in range(n_clusters)]
        distances = cdist(centroids, xs, metric='sqeuclidean')
        idxs = np.argmin(distances, axis=0)
        assert idxs.shape[0] == n_points
        centroids_old = np.copy(centroids)
        for i in range(n_clusters):
            idxs_i = idxs == i
            if idxs_i.any():
                centroids[i] = np.mean(xs[idxs_i], axis=0)

    return centroids

def mapper(key, value):
    # key: None
    # value: Numpy array of the points.
    n_clusters = 1000
    uniform_probs = np.ones(value.shape[0]) / value.shape[0]
    centers = sample_from_points(value, p=uniform_probs, n_samples=n_clusters)
    for i in range(2):
        centers = d3sample(value, centers, n_clusters=n_clusters)
    yield 0, centers
    print('map')

def reducer(key, values):
    # key: key from mapper used to aggregate (constant)
    # values: list of cluster centers.

    batch_centroids = np.reshape(values, (-1, NUM_FEATURES))

    centers = kmeans_mem(batch_centroids)
    print('ex')
    yield centers
