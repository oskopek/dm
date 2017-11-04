import numpy as np
import random

# Fix random seed.
np.random.seed(42)
random.seed(42)

# Feature constants.
features = 400
FEATURE_IDS = range(features)


# Polynomial kernel with d = 2, c = 0 for (x^T * y + c)^d
# Dtype trick: float16 for evaluation, float32 for training (memory limit).
def transform(X, dtype=np.float16):
    # Make sure this function works for both 1D and 2D NumPy arrays.
    if len(X.shape) == 1:
        X = np.reshape(X, (1, -1))

    n_features = X.shape[1]
    sqrt2 = np.sqrt(2)
    normalizer = np.ones((n_features, n_features)) - \
        np.eye(n_features) * (1 - (1 / sqrt2))
    Xnew = np.zeros((X.shape[0], n_features**2), dtype=dtype)
    for i, row in enumerate(X):
        rrT = sqrt2 * np.outer(row, row)  # row * row^T
        rrT *= normalizer  # element by element, divides diagonal by sqrt(2)
        Xnew[i, :] = rrT.flat
    return Xnew


# ADAM with the OCP SVM.
def ocp_svm_adam(X, y, lambada=1, beta1=0.9, beta2=0.999):
    EPS = 0.000001

    m_t_prev = 0  # last 1st moment
    v_t_prev = 0  # last 2nd moment
    w = np.zeros((features**2,))
    for t, row in enumerate(X):
        row = transform(row, dtype=np.float32)
        row = np.reshape(row, (features**2))
        lr_t = lambada * np.sqrt(1 - beta2**(t + 1)) / (1 - beta1**(t + 1))
        g_t = -y[t] * row
        m_t = beta1 * m_t_prev + (1 - beta1) * g_t
        v_t = beta2 * v_t_prev + (1 - beta2) * (g_t * g_t)
        m_hat_t = m_t / (1 - beta1**(t + 1))
        v_hat_t = v_t / (1 - beta2**(t + 1))
        if y[t] * np.dot(row, w) < 1:
            w -= lr_t * m_hat_t / (np.sqrt(v_hat_t) + EPS)

        m_t_prev = m_t
        v_t_prev = v_t
    return w


# SGD with OCP SVM.
# Not used.
def ocp_svm_sgd(X, y, lambada=1):
    eta_mult = 5
    inv_lambda = 1 / lambada

    w = np.zeros((features**2,))
    for t, row in enumerate(X):
        row = transform(row, dtype=np.float32)
        row = np.reshape(row, (features**2))
        eta = eta_mult * 1 / np.sqrt(t + 1)
        if y[t] * np.dot(row, w) < 1:
            # w = w - eta * (-y[t] * row)
            w += eta * y[t] * row
            w *= min(1, inv_lambda / np.linalg.norm(w))
    return w


# Shuffles and parses the minibatch, and runs OCP SVM (SGD/ADAM).
def mapper(key, value):
    # key: None
    # value: one line of input file
    np.random.shuffle(value)
    value = np.array([map(np.float32, row.split()) for row in value])
    X = value[:, 1:]
    y = value[:, 0]
    # yield 0, ocp_svm_sgd(X, y)
    yield 0, ocp_svm_adam(X, y)


# Input: Single key, sorted weight vectors
# Output: Weight vector for prediction, averaged
def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    yield np.mean(values, axis=0)
