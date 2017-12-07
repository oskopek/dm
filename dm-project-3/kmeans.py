from __future__ import division, print_function
import numpy as np
from scipy.spatial.distance import cdist

# The number of preprocessed features in the input.
NUM_FEATURES = 250

# Set the random seed
np.random.seed(42)


# Calculate the squared distances, rows represent distances from centers,
# columns distances from points in xs.
def sqdist(centers, xs):
    # TODO: Try a different metric
    return cdist(centers, xs, metric='sqeuclidean')


# Sample n_samples points from xs according to the distribution p,
# without replacement.
def sample_from_points(xs, p, n_samples):
    assert(n_samples <= xs.shape[0])
    idx = np.random.choice(xs.shape[0], n_samples, replace=False, p=p)
    return xs[idx], 1/p[idx]


# Sample n_samples points from xs based on the inverse squared distance
# to centers.
# Algorithm 1: https://arxiv.org/pdf/1703.06476.pdf
def d2sample(xs, centers, n_clusters=200):
    distances = sqdist(centers, xs)
    min_distances = np.min(distances, axis=0)  # from points to centers
    probs = min_distances / np.sum(min_distances)
    return sample_from_points(xs, probs, n_clusters)


# Implements coreset construction by iterative sampling.
# Algorithm 2: https://arxiv.org/pdf/1703.06476.pdf
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
def kmeans_pp(xs, n_clusters, weights):
    # Sample a new point based on the inverse squared distance to centers,
    # iteratively; uses distances as a matrix that it fills in top to bottom.
    def kpp_sample_iter(xs, distances, iteration):
        min_distances = np.min(distances[:iteration], axis=0)
        probs = min_distances / np.sum(min_distances)
        new_point, _ = sample_from_points(xs, probs, 1)
        distances[iteration, :] = sqdist(new_point, xs)
        return new_point

    # Initial distribution.
    n_points = xs.shape[0]
    probs = np.ones(n_points) / n_points

    centers = np.zeros((n_clusters, NUM_FEATURES))
    distances = np.zeros((n_clusters, n_points))
    centers[0], _ = sample_from_points(xs, probs, 1)
    distances[0] = sqdist([centers[0]], xs) * weights
    for k in range(1, n_clusters):
        centers[k] = kpp_sample_iter(xs, distances, k)
    return centers


# Implements kmeans with all points and distances in memory.
def kmeans_mem(xs, initial_centers=None, n_clusters=200, n_iterations=50,
               early_stopping=0.2, weights=1):
    # Run kmeans++ if initial_centers were not provided.
    if initial_centers is None:
        centers = kmeans_pp(xs, n_clusters, weights)
        print('kmeans++:\tend')
    else:
        centers = initial_centers

    centers_old = np.zeros_like(centers)
    for iteration in range(n_iterations):
        # Early stopping based on the norm of the change in centers.
        diff = np.linalg.norm(centers - centers_old)
        print('kmeans_mem:\tit:\t{}\tdiff:\t{}'.format(iteration, diff))
        if diff < early_stopping:
            break

        distances = sqdist(centers, xs) * weights
        idxs = np.argmin(distances, axis=0)
        assert idxs.shape[0] == xs.shape[0]
        centers_old = np.copy(centers)
        for i in range(n_clusters):
            idxs_i = idxs == i
            if idxs_i.any():
                centers[i] = np.mean(xs[idxs_i], axis=0)

    return centers


# Mapper: runs d3sample on uniformly sampled centers and yields them.
def mapper(key, value):
    # key: None
    # value: Numpy array of the points.
    n_points = value.shape[0]
    n_clusters = min(n_points-1, 3000)
    uniform_probs = np.ones(n_points) / n_points
    centers, _ = sample_from_points(value, p=uniform_probs, n_samples=n_clusters)
    centers, weights = d3sample(value, centers, n_clusters=n_clusters)
    print('mapper:\tend')
    # A 'separator' value is used in order for pickle not to squash everything
    # into a single numpy array.
    yield 0, (centers, weights, 'separator', value)


# Reducer: runs kmeans_mem on the centers provided (includes KMeans++).
def reducer(key, values):
    # key: key from mapper used to aggregate (constant)
    # values: list of cluster centers.
    batch_centers = np.asarray([value[0] for value in values])
    batch_centers = np.reshape(batch_centers, (-1, NUM_FEATURES))
    centers_weights = np.asarray([value[1] for value in values])
    centers_weights = np.reshape(centers_weights, (-1))
    all_pts = np.asarray([value[3] for value in values])
    all_pts = np.reshape(all_pts, (-1, NUM_FEATURES))

    centers = kmeans_mem(batch_centers, weights=centers_weights,
                         early_stopping=0.09)
    centers = kmeans_mem(all_pts, initial_centers=centers, early_stopping=0.5)
    print('reducer:\tend')
    yield centers
