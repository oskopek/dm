from __future__ import division
import numpy as np
import random
from scipy.spatial.distance import cdist

NUM_FEATURES = 250

random.seed(42)
np.random.seed(42)

def sample_from_points(xs, p, n_samples, output_weights=False):
    assert(n_samples <= xs.shape[0])
    idx = np.random.choice(xs.shape[0], n_samples, replace=False, p=p)
    if output_weights:
        return xs[idx], 1/p[idx]
    else:
        return xs[idx]

def d2sample(xs, centroids, n_clusters=200):
    distances = cdist(centroids, xs, metric='sqeuclidean')
    min_distances = np.min(distances, axis=0)
    probs = min_distances / np.sum(min_distances)
    return sample_from_points(xs, probs, n_clusters)

def kpp_sample_iter(xs, distances, iteration):
    min_distances = np.min(distances[:iteration], axis=0)
    probs = min_distances / np.sum(min_distances)
    new_point = sample_from_points(xs, probs, 1)
    distances[iteration,:] = cdist(new_point, xs, metric='sqeuclidean')
    return new_point

def d3sample(xs, centroids, n_clusters=200):
    n_points = xs.shape[0]
    # alpha = np.log2(n_clusters)+1
    alpha = 16 * (np.log2(n_clusters) + 2)
    distances = cdist(centroids, xs, metric='sqeuclidean')
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
    sum_of_dists_in_cluster = np.vectorize(lambda i: np.sum(min_distances[B_is==i]))(range(n_clusters))

    probs  = alpha*min_distances/c_phi + \
             2*alpha*sum_of_dists_in_cluster[B_is] / (B_i_lens * c_phi) + \
             4 * n_points/B_i_lens

    probs = probs / np.sum(probs)
    return sample_from_points(xs, probs, n_clusters, True)

def kmeans_pp(xs, n_clusters, weights):
    n_points = xs.shape[0]
    probs = np.ones(n_points)/n_points
    centroids = np.zeros((n_clusters, NUM_FEATURES))
    centroids[0,:] = sample_from_points(xs, probs, 1)
    distances = np.zeros((n_clusters, n_points))
    distances[0,:] = cdist([centroids[0,:]], xs, metric='sqeuclidean') * weights
    for k in range(1, n_clusters):
        centroids[k,:] = kpp_sample_iter(xs, distances, k)
    return centroids

def kmeans_mem(xs, initial_centroids=None, n_clusters=200, n_iterations=50, early_stopping=0.2, weights=1):
    n_points = xs.shape[0]
    if initial_centroids is None:
        centroids = kmeans_pp(xs, n_clusters, weights)
        print('kpp')
    else:
        centroids = initial_centroids

    centroids_old = np.zeros_like(centroids)
    for iteration in range(n_iterations):
        diff = np.linalg.norm(centroids-centroids_old)
        print 'it', iteration, ' diff: ', diff
        if diff < early_stopping:
            break
        distances = cdist(centroids, xs, metric='sqeuclidean') * weights
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
    n_points = value.shape[0]
    n_clusters = min(n_points-1, 3000)
    uniform_probs = np.ones(value.shape[0]) / value.shape[0]
    centers = sample_from_points(value, p=uniform_probs, n_samples=n_clusters)
    centers, weights = d3sample(value, centers, n_clusters=n_clusters)
    print('map')
    yield 0, (centers, weights, "separator", value)

# Not used
def eval_k_means(xs, centers):
    distances = cdist(centers, xs, metric='sqeuclidean')
    min_distances = np.min(distances, axis=0)
    return np.mean(min_distances)

def reducer(key, values):
    # key: key from mapper used to aggregate (constant)
    # values: list of cluster centers.
    batch_centroids = np.asarray([value[0] for value in values])
    batch_centroids = np.reshape(batch_centroids, (-1, NUM_FEATURES))
    centroids_weights = np.asarray([value[1] for value in values])
    centroids_weights = np.reshape(centroids_weights, (-1))
    all_pts = np.asarray([value[3] for value in values])
    all_pts = np.reshape(all_pts, (-1, NUM_FEATURES))

    centers = kmeans_mem(batch_centroids, weights=centroids_weights, early_stopping=0.09)
    centers = kmeans_mem(all_pts, initial_centroids=centers, early_stopping=0.5)
    print('ex')
    yield centers
