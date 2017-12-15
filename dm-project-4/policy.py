import numpy as np

USER_FEATURES = 6
ARTICLE_FEATURES = 12

#DELTA = 0.1
#ALPHA = 1.0 + np.sqrt(np.log(2.0/DELTA)/2.0)
ALPHA = 0.7
print(ALPHA)

A0 = np.eye(USER_FEATURES)
A0inv = np.eye(USER_FEATURES)
b0 = np.zeros(USER_FEATURES)

class CL:
    def __init__(self, id, x, A, Ainv, B, b):
        self.id = id
        self.x = x
        self.A = A
        self.Ainv = Ainv
        self.B = B
        self.b = b

CLS = dict()

last_chosen_cl = None
last_z = None
last_x = None


def set_articles(articles_local):
    global articles
    articles = articles_local
    # articles - dictionary of (about 80) article id -> features (of len 6)
    for article_id, article in articles.iteritems():
        CLS[article_id] = CL(
                id = article_id,
                x = np.asarray(article),
                A = np.eye(ARTICLE_FEATURES),
                Ainv = np.eye(ARTICLE_FEATURES),
                B = np.zeros((ARTICLE_FEATURES, USER_FEATURES)),
                b = np.zeros(ARTICLE_FEATURES))


def update(reward):
    # reward - int
    if reward == -1:
        return

    global last_chosen_cl, last_z, A0, b0, A0inv, last_x
    a = last_chosen_cl

    dott = a.B.T.dot(a.Ainv)
    np.add(A0, dott.dot(a.B), out=A0)
    np.add(b0, dott.dot(a.b), out=b0)

    np.add(a.A, np.outer(last_x, last_x), out=a.A)
    a.Ainv =  np.linalg.inv(a.A)
    np.add(a.B, np.outer(last_x, last_z), out=a.B)
    np.add(a.b, reward * last_x, out=a.b)

    dott = a.B.T.dot(a.Ainv)
    np.add(A0, np.outer(last_z, last_z) - dott.dot(a.B), out=A0)
    A0inv = np.linalg.inv(A0)
    np.add(b0, reward * last_z - dott.dot(a.b), out=b0)


def recommend(time, user_features, choices):
    # time - int
    # user_features - list - user features, len 6
    # choices - list - ids of articles to choose from, len 20
    global last_chosen_cl, last_z, last_x
    z = np.asarray(user_features)
    x = np.zeros((ARTICLE_FEATURES))
    x[USER_FEATURES:] = z
    beta = A0inv.dot(b0)

    ps = np.zeros(len(choices))
    for i, article_id in enumerate(choices):
        a = CLS[article_id]
        x[:USER_FEATURES] = a.x
        theta = a.Ainv.dot(a.b - a.B.dot(beta))

        Ainvx = a.Ainv.dot(x)
        middle_vector = A0inv.dot(a.B.T.dot(Ainvx))
        s = z.dot(A0inv.dot(z)) - 2 * z.dot(middle_vector) + x.dot(Ainvx)\
            + x.dot(a.Ainv.dot(a.B.dot(middle_vector)))

        p = z.dot(beta) + x.dot(theta) + ALPHA * np.sqrt(s)
        ps[i] = p

    p_argmax_id = choices[np.argmax(ps)]
    p_argmax = CLS[p_argmax_id]
    x[:USER_FEATURES] = p_argmax.x
    last_x = x
    last_chosen_cl = p_argmax
    last_z = z
    return p_argmax.id
