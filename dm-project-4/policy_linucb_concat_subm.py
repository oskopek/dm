import numpy as np

USER_FEATURES = 6
ARTICLE_FEATURES = 6
FEATURES = ARTICLE_FEATURES + USER_FEATURES

last_chosen_idx = None
last_z = None

ALPHA = 0.18627451

indexes = dict()
inv_indexes = dict()


def a_index(article_ids):
    if isinstance(article_ids, int):
        return indexes[article_ids]
    else:
        return [indexes[idx] for idx in article_ids]


def set_articles(articles, alpha=None):
    # articles - dictionary of (about 80) article id -> features (of len 6)
    counter = len(articles)
    global xs, Ms, Minvs, bs, ALPHA
    if alpha is not None:
        ALPHA = float(alpha)
    print(ALPHA)

    xs = np.zeros((counter, ARTICLE_FEATURES, 1))
    Ms = np.zeros((counter, FEATURES, FEATURES))
    Minvs = np.zeros((counter, FEATURES, FEATURES))
    bs = np.zeros((counter, FEATURES, 1))

    counter = 0
    for article_id, article in articles.iteritems():
        indexes[article_id] = counter
        inv_indexes[counter] = article_id

        xs[counter, :, 0] = np.asarray(article)
        Ms[counter] = np.eye(FEATURES)
        Minvs[counter] = np.eye(FEATURES)
        bs[counter] = np.zeros((FEATURES, 1))

        counter += 1


def update(reward):
    # reward - int
    if reward == -1:
        return
    # TODO
    #elif reward == 0:
    #    reward = -1

    global last_chosen_idx, last_z
    Ms[last_chosen_idx] += np.matmul(last_z, last_z.T)
    Minvs[last_chosen_idx] = np.linalg.inv(Ms[last_chosen_idx])
    bs[last_chosen_idx] += reward * last_z


def recommend(time, user_features, choices):
    # time - int
    # user_features - list - user features, len 6
    # choices - list - ids of articles to choose from, len 20
    global last_chosen_idx, last_z
    n_choices = len(choices)
    user = np.asarray(user_features)

    idx = a_index(choices)  # (n_choices)
    Minv = Minvs[idx]  # (n_choices, ARTICLE, ARTICLE)
    b = bs[idx]  # (n_choices, ARTICLE, 1)

    user = np.expand_dims(user, 0)
    user = np.repeat(user, n_choices, axis=0)
    user = np.expand_dims(user, -1)
    # (n_choices, USER_FEATURES, 1)

    xss = xs[idx]
    z = np.zeros((n_choices, FEATURES, 1))
    z[:,:ARTICLE_FEATURES,:] = user
    z[:,ARTICLE_FEATURES:,:] = xss

    #z = user
    zT = np.transpose(z, axes=(0, 2, 1))

    w = np.matmul(Minv, b)  # (n_choices, ARTICLE, 1)
    cov = np.matmul(zT, np.matmul(Minv, z))
    wT = np.transpose(w, axes=(0, 2, 1))
    UCB = np.matmul(wT, z) + ALPHA * np.sqrt(cov)

    amax = np.argmax(UCB)
    last_chosen_idx = idx[amax]
    last_z = z[amax]
    return choices[amax]
