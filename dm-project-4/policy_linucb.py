import numpy as np
import random

USER_FEATURES = 6
ARTICLE_FEATURES = 6
FEATURES = USER_FEATURES + ARTICLE_FEATURES

DELTA = 0.05
ALPHA = 0.2 # 1 + np.sqrt(np.log(2/DELTA)/2)
print(ALPHA)

garticles = None

random.seed(42)
np.random.seed(42)

M = dict()
Minvs = dict()
b = dict()

last_chosen_article_id = -1
last_zt = None


def set_articles(articles):
    # articles - dictionary of (about 80) article id -> features (of len 6)
    global garticles
    garticles = articles
    for article_id in articles.keys():
        garticles[article_id] = np.asarray(articles[article_id])
        M[article_id] = np.eye(FEATURES)
        Minvs[article_id] = np.eye(FEATURES)
        b[article_id] = np.zeros(FEATURES)

def update(reward):
    # reward - int
    if reward == -1:
        return
    global last_chosen_article_id, last_zt
    assert last_chosen_article_id >= 0
    assert last_zt is not None
    article = last_chosen_article_id
    M[article] += np.outer(last_zt, last_zt)
    Minvs[article] = np.linalg.inv(M[article])
    b[article] += reward * last_zt

    last_zt = None
    last_chosen_article_id = -1


def recommend(time, user_features, choices):
    # time - int
    # user_features - list - user features, len 6
    # choices - list - ids of articles to choose from, len 20
    zt = np.zeros(FEATURES)
    zt[:USER_FEATURES] = np.asarray(user_features)
    global garticles, last_zt

    UCB_max = np.NINF
    UCB_argmax = np.random.choice(choices)
    for article_id in choices:
        zt[USER_FEATURES:FEATURES] = garticles[article_id]
        Minv = Minvs[article_id]
        w = Minv.dot(b[article_id])
        UCB = w.dot(zt) + ALPHA * np.sqrt(zt.dot(Minv.dot(zt)))
        if UCB > UCB_max:
            last_zt = np.copy(zt)
            UCB_max = np.copy(UCB)
            UCB_argmax = article_id

    global last_chosen_article_id
    last_chosen_article_id = UCB_argmax
    #last_zt = zt
    return UCB_argmax
