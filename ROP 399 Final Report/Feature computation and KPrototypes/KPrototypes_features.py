import numpy as np
import pandas as pd
import os
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from kmodes.kprototypes import KPrototypes
'''
Path to ROI features
'''
features_loc = 'DBSCAN/Newfeatures.csv'

'''
Path to K Means clustering output
'''
KMeans_output = 'C:/Users/MedImage7271/Desktop/PyCharm Projects/Dental plate artifact classification/DBSCAN/KMeans-scaled/'

if not os.path.isdir(KMeans_output):
    os.makedirs(KMeans_output)

'''
Number of K Means clusters
'''
num_clusters = 5



features = pd.read_csv(features_loc, index_col=0)
i, h = features.index, features.columns
f = features.drop(['Shape'], axis = 1)
shape = features['Shape']
scaler = MinMaxScaler()
f = pd.DataFrame(scaler.fit_transform(f), index=i, columns=h[:-1])
a = pd.concat([f,shape], axis = 1)
kmeans = KPrototypes(n_clusters=num_clusters, init='Cao', verbose=2).fit(a, categorical=[8])
# kmeans = KMeans(n_clusters=num_clusters, random_state=384).fit(features)
summary = pd.DataFrame(kmeans.labels_, index=features.index)
summary.to_csv(KMeans_output + 'clusters.csv', header=False)
print(summary.apply(pd.value_counts))