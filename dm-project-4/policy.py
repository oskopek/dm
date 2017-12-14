import numpy as np

USER_FEATURES = 6
ARTICLE_FEATURES = 6

DELTA = 0.05
ALPHA = 1 + np.sqrt(np.log(2/DELTA)/2)
print(ALPHA)

M = dict()
b = dict()

w = dict()
UCB = dict()

last_chosen_article_id = -1
last_zt = None


def set_articles(articles):
    # articles - dictionary of (about 80) article id -> features (of len 6)
    for article_id in articles.keys():
        M[article_id] = np.eye(USER_FEATURES, USER_FEATURES)
        b[article_id] = np.zeros(USER_FEATURES)


def update(reward):
    # reward - int
    global last_chosen_article_id, last_zt
    assert last_chosen_article_id >= 0
    assert last_zt is not None
    article = last_chosen_article_id
    M[article] += np.outer(last_zt, last_zt)
    b[article] += reward * last_zt

    last_zt = None
    last_chosen_article_id = -1


def recommend(time, user_features, choices):
    # time - int
    # user_features - list - user features, len 6
    # choices - list - ids of articles to choose from, len 20
    zt = np.asarray(user_features)

    UCB_max = np.NINF
    UCB_argmax = np.random.choice(choices)
    for article_id in choices:
        Minv = np.linalg.inv(M[article_id])
        w[article_id] = np.matmul(Minv, b[article_id])
        UCB[article_id] = np.dot(w[article_id], zt) + ALPHA * np.sqrt(np.dot(zt, np.matmul(Minv, zt)))
        if UCB[article_id] > UCB_max:
            UCB_max = UCB[article_id]
            UCB_argmax = article_id

    global last_chosen_article_id, last_zt
    last_chosen_article_id = UCB_argmax
    last_zt = zt
    return UCB_argmax
