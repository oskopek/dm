import numpy as np
import random

np.random.seed(42)
random.seed(42)

FEATURES = 400


def transform(X):
    # Make sure this function works for both 1D and 2D NumPy arrays.
    return X

def ocp_svm(X, y, lambada=1):
    eta_mult = 5
    inv_lambda = 1 / lambada

    m = np.zeros((FEATURES)) # 1st moment
    v = np.zeros((FEATURES)) # 2nd moment
    w = np.zeros((FEATURES))
    for t, row in enumerate(X):
        eta = eta_mult * 1 / np.sqrt(t+1)
        if y[t] * np.dot(row, w) < 1:
            #w = w - eta * (-y[t] * row)
            w += eta * y[t] * row
            w *= min(1, inv_lambda / np.linalg.norm(w))
    return w


def mapper(key, value):
    # key: None
    # value: one line of input file
    np.random.shuffle(value)
    value = np.array([map(float, row.split()) for row in value])
    X = value[:, 1:]
    y = value[:, 0]

    

    yield 0, ocp_svm(X, y, lambada=1)


# Input: Single key, sorted weight vectors
# Output: Weight vector for prediction
def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    w = np.zeros((FEATURES))
    for w_i in values:
        w += w_i
    w /= len(values)
    yield w


