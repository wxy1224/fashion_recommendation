from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import cv2

features = np.load('demo/output/features.npy')
print('num of images = ', len(features))
print('features loaded!')

feature_wenxin = np.load('demo/output/features_wenxin.npy')

def find_knn(k = 6):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', n_jobs=-1).fit(features[:, :])
    distances, indices = nbrs.kneighbors(feature_wenxin)
    print('First half done...')
    distances, a = nbrs.kneighbors(features[60000:61000, 1:])
    indices = np.concatenate((indices, a))
    print(indices[0:5])
    np.save('demo/output/indices_wenxin.npy', indices)


find_knn()

imgs = pd.read_csv('demo/full_data_revised.csv', usecols=['image_path']).values
idx = np.load('demo/output/indices_wenxin.npy')

for i in range(1910, 1915):
    for j in idx[i, :]:
        img = cv2.imread(imgs[j, 0])
        cv2.imshow('image', img)
        cv2.waitKey(0)

