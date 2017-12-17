import numpy as np

USER_FEATURES = 6
ARTICLE_FEATURES = 6
OUT_FEATURES = ARTICLE_FEATURES * USER_FEATURES

# DELTA = 0.01
# ALPHA = 1.0 + np.sqrt(np.log(2.0/DELTA)/2.0)
ALPHA = 0.2
print(ALPHA)

A0 = np.eye(OUT_FEATURES)
A0inv = np.eye(OUT_FEATURES)
b0 = np.zeros((OUT_FEATURES, 1))

UPPER_ARTICLES = 271

xs = np.zeros((UPPER_ARTICLES, ARTICLE_FEATURES, 1))
As = np.zeros((UPPER_ARTICLES, USER_FEATURES, USER_FEATURES))
Ainvs = np.zeros((UPPER_ARTICLES, USER_FEATURES, USER_FEATURES))
Bs = np.zeros((UPPER_ARTICLES, USER_FEATURES, OUT_FEATURES))
bs = np.zeros((UPPER_ARTICLES, USER_FEATURES, 1))

last_chosen_idx = None
last_z = None
last_x = None

indexes = dict()
inv_indexes = dict()


def a_index(article_ids):
    if isinstance(article_ids, int):
        return indexes[article_ids]
    else:
        return [indexes[idx] for idx in article_ids]


def set_articles(articles):
    # articles - dictionary of (about 80) article id -> features (of len 6)
    counter = 0
    for article_id, article in articles.iteritems():
        indexes[article_id] = counter
        inv_indexes[counter] = article_id

        xs[counter, :, 0] = np.asarray(article)
        As[counter] = np.eye(USER_FEATURES)
        Ainvs[counter] = np.eye(USER_FEATURES)
        Bs[counter] = np.zeros((USER_FEATURES, OUT_FEATURES))
        bs[counter] = np.zeros((USER_FEATURES, 1))

        counter += 1


def update(reward):
    # reward - int
    if reward == -1:
        return
    elif reward == 0:
        reward = -1

    global last_chosen_idx, last_z, A0, b0, A0inv, last_x
    A = As[last_chosen_idx]  # (ARTICLE, ARTICLE)
    Ainv = Ainvs[last_chosen_idx]  # (ARTICLE, ARTICLE)
    B = Bs[last_chosen_idx]  # (ARTICLE, USER)
    b = bs[last_chosen_idx]  # (ARTICLE, 1)

    dott = np.matmul(B.T, Ainv)
    np.add(A0, np.matmul(dott, B), out=A0)
    np.add(b0, np.matmul(dott, b), out=b0)

    last_zT = last_z.T
    np.add(A, np.matmul(last_x, last_x.T), out=A)
    As[last_chosen_idx] = A
    Ainv = np.linalg.inv(A)
    Ainvs[last_chosen_idx] = Ainv
    np.add(B, np.matmul(last_x, last_zT), out=B)
    Bs[last_chosen_idx] = B
    np.add(b, reward * last_x, out=b)
    bs[last_chosen_idx] = b

    dott = np.matmul(B.T, Ainv)
    np.add(A0, np.matmul(last_z, last_zT) - np.matmul(dott, B), out=A0)
    A0inv = np.linalg.inv(A0)
    np.add(b0, reward * last_z - np.matmul(dott, b), out=b0)


def recommend(time, user_features, choices):
    # time - int
    # user_features - list - user features, len 6
    # choices - list - ids of articles to choose from, len 20
    global last_chosen_idx, last_z, last_x
    n_choices = len(choices)
    user = np.asarray(user_features)

    idx = a_index(choices)  # (n_choices)
    Ainv = Ainvs[idx]  # (n_choices, ARTICLE, ARTICLE)
    B = Bs[idx]  # (n_choices, ARTICLE, USER)
    b = bs[idx]  # (n_choices, ARTICLE, 1)

    beta = np.matmul(A0inv, b0)  # (ARTICLE, 1)

    user = np.expand_dims(user, 0)
    user = np.repeat(user, n_choices, axis=0)
    user = np.expand_dims(user, -1)
    # (n_choices, USER_FEATURES, 1)

    xss = xs[idx]
    xsT = np.transpose(xss, axes=(0, 2, 1))
    x = np.matmul(user, xsT)
    x = np.reshape(x, (n_choices, OUT_FEATURES, 1))
    z = x
    x = user

    xT = np.transpose(x, axes=(0, 2, 1))
    zT = np.transpose(z, axes=(0, 2, 1))

    thetas = np.matmul(Ainv, b - np.matmul(B, beta))  # (n_choices, ARTICLE, 1)
    Ainvxs = np.matmul(Ainv, x)  # (n_choices, ARTICLE, 1)
    BT = np.transpose(B, axes=(0, 2, 1))  # (n_choices, USER, ARTICLES)

    # (n_choices, USER, ARTICLES)
    middle_vectors = np.matmul(A0inv, np.matmul(BT, Ainvxs))
    ss = np.matmul(zT, np.matmul(A0inv, z))  # (n_choices, 1, 1)
    ss -= 2 * np.matmul(zT, middle_vectors)
    ss += np.matmul(xT, Ainvxs)
    ss += np.matmul(xT, np.matmul(Ainv, np.matmul(B, middle_vectors)))
    ps = np.matmul(zT, beta) + np.matmul(xT, thetas) + ALPHA * np.sqrt(ss)
    ps = np.squeeze(ps)

    amax = np.argmax(ps)
    last_chosen_idx = idx[amax]
    last_z = z[amax]
    last_x = x[amax]
    return choices[amax]
