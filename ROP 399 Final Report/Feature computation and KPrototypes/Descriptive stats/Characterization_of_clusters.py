import pandas as pd
import os

cluster_loc = 'DBSCAN/KMeans-scaled/clusters.csv'
features_loc = 'DBSCAN/Newfeatures.csv'
out_loc = 'DBSCAN/Characterization/'
if not os.path.isdir(out_loc):
    os.makedirs(out_loc)

clusters = pd.read_csv(cluster_loc, index_col=0, header=None)
features = pd.read_csv(features_loc, index_col=0)

clusters.columns = ['Cluster']
f = pd.concat([features, clusters], axis=1)
groupby = f.groupby(['Cluster']).median()
print(groupby)
groupby.to_csv(out_loc + 'characterization_of_clusters2.csv')