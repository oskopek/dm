import numpy as np

USER_FEATURES = 6
ARTICLE_FEATURES = 6

DELTA = 0.1
ALPHA = 1.0 + np.sqrt(np.log(2.0/DELTA)/2.0)
print(ALPHA)

articles = None

A0 = np.eye(USER_FEATURES)
A0inv = np.eye(USER_FEATURES)
b0 = np.zeros(USER_FEATURES)

A = dict()
Ainvs = dict()
B = dict()
b = dict()

last_chosen_article_id = -1
last_zt = None


def set_articles(articles_local):
    global articles
    articles = articles_local
    # articles - dictionary of (about 80) article id -> features (of len 6)
    for article_id, article in articles.iteritems():
        articles[article_id] = np.asarray(article)
        A[article_id] = np.eye(ARTICLE_FEATURES, ARTICLE_FEATURES)
        Ainvs[article_id] = np.eye(ARTICLE_FEATURES, ARTICLE_FEATURES)
        B[article_id] = np.zeros((ARTICLE_FEATURES, USER_FEATURES))
        b[article_id] = np.zeros(ARTICLE_FEATURES)


def update(reward):
    # reward - int
    global last_chosen_article_id, last_zt, A0, b0, A0inv
    assert last_chosen_article_id >= 0
    assert last_zt is not None
    article_id = last_chosen_article_id
    zt = last_zt

    if reward == -1:
        return

    dott = B[article_id].T.dot(Ainvs[article_id])
    A0 += dott.dot(B[article_id])
    b0 += dott.dot(b[article_id])

    x = articles[article_id]
    A[article_id] += np.outer(x, x)
    Ainvs[article_id] = np.linalg.inv(A[article_id])
    B[article_id] += np.outer(x, zt)
    b[article_id] += reward * x

    dott = B[article_id].T.dot(Ainvs[article_id])
    A0 += np.outer(zt, zt) - dott.dot(B[article_id])
    A0inv = np.linalg.inv(A0)
    b0 += reward * zt - dott.dot(b[article_id])


def recommend(time, user_features, choices):
    # time - int
    # user_features - list - user features, len 6
    # choices - list - ids of articles to choose from, len 20
    zt = np.asarray(user_features)
    beta = np.dot(A0inv, b0)

    p_max = np.NINF
    p_argmax = np.random.choice(choices)
    for article_id in choices:
        Ainv = Ainvs[article_id]
        x = articles[article_id]
        theta = Ainv.dot(b[article_id] - B[article_id].dot(beta))

        middle_vector = A0inv.dot(B[article_id].T.dot(Ainv.dot(x)))
        s = zt.dot(A0inv.dot(zt)) - 2 * zt.dot(middle_vector) + x.dot(Ainv.dot(x))\
            + x.dot(Ainv.dot(B[article_id].dot(middle_vector)))

        p = zt.dot(beta) + x.dot(theta) + ALPHA * np.sqrt(s)
        if p > p_max:
            p_max = p
            p_argmax = article_id

    global last_chosen_article_id, last_zt
    last_chosen_article_id = p_argmax
    last_zt = zt
    return p_argmax
