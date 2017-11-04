import numpy as np
import random

np.random.seed(42)
random.seed(42)

FEATURE_IDS = np.random.choice(range(400), 320, replace=False)

def transform(X):
    # Make sure this function works for both 1D and 2D NumPy arrays.
    X1 = X[:, FEATURE_IDS]
    X2 = np.zeros((X1.shape[0], X1.shape[1]**2), dtype=np.float16)
    sqrt2 = np.sqrt(2)
    for i, row in enumerate(X1):
        Xn = sqrt2 * np.outer(row, row)
        Xn *= np.ones((len(row), len(row))) - np.eye(len(row)) * (1 - 1/sqrt2)
        X2[i, :] = Xn.flat
    #return np.concatenate((X, X2), axis=1)
    return X2

def ocp_svm(X, y, lambada=10, beta1=0.9, beta2=0.999):
    print("Training batch")
    EPS = 0.000001

    m_t_prev = 0 # last 1st moment
    v_t_prev = 0 # last 2nd moment
    w = np.zeros((X.shape[1],))
    for t, row in enumerate(X):
        row = row.astype(np.float32)
        lr_t = lambada * np.sqrt(1 - beta2**(t+1)) / (1 - beta1**(t+1))
        g_t = -y[t] * row
        m_t = beta1 * m_t_prev + (1 - beta1) * g_t
        v_t = beta2 * v_t_prev + (1 - beta2) * (g_t * g_t)
        m_hat_t = m_t / (1 - beta1**(t+1))
        v_hat_t = v_t / (1 - beta2**(t+1))
        if y[t] * np.dot(row, w) < 1:
            w -= lr_t * m_hat_t / (np.sqrt(v_hat_t) + EPS)

        m_t_prev = m_t
        v_t_prev = v_t
    return w


def mapper(key, value):
    # key: None
    # value: one line of input file
    np.random.shuffle(value)
    value = np.array([map(np.float32, row.split()) for row in value])
    X = value[:, 1:]
    y = value[:, 0]
    X = transform(X)
    yield 0, ocp_svm(X, y)


# Input: Single key, sorted weight vectors
# Output: Weight vector for prediction
def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    w = np.zeros((len(values[0]), ))
    for w_i in values:
        w += w_i
    w /= len(values)
    yield w


