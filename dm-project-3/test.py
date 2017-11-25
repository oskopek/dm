import numpy as np
train_data = np.load('data/handout_train.npy')
test_data = np.load('data/handout_test.npy')

from sklearn.cluster import KMeans
clf = KMeans(n_clusters=200)
clf.fit(train_data)

predicted = clf.predict(test_data)
print(predicted)
