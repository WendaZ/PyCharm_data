import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

cluster_loc = 'DBSCAN/KMeans-scaled/clusters.csv'
features_loc = 'DBSCAN/Newfeatures.csv'
out_loc = 'DBSCAN/Characterization/Hist/'
if not os.path.isdir(out_loc):
    os.makedirs(out_loc)

clusters = pd.read_csv(cluster_loc, index_col=0, header=None)
features = pd.read_csv(features_loc, index_col=0)
clusters.columns = ['Cluster']
f = pd.concat([features, clusters], axis=1)
cluster = sorted(f['Cluster'].unique().tolist())
# groupby = f.groupby(['Cluster']).median()
# print(groupby)
# groupby.to_csv(out_loc + 'characterization_of_clusters2.csv')
# f.to_csv(out_loc + 'summary.csv')
for i in range(len(f.columns)-1):
    a = pd.concat([f.iloc[:, i], f.iloc[:, -1]], axis=1)
    # a.plot.scatter(x=a.columns[-1], y=a.columns[0])
    # plt.savefig(out_loc + a.columns[0] + '.png')
    for j in cluster:
        b = a[a['Cluster'] == j]
        b.iloc[:, 0].plot.hist(bins=np.linspace(min(b.iloc[:, 0]), max(b.iloc[:, 0]),100) ,rwidth=0.5 )
        plt.xlabel("mean = " + str(b.iloc[:, 0].mean()) + "    variance = " + str(b.iloc[:, 0].var()))
        plt.title("Cluster " + str(j) + ' ' + f.columns[i] + ' distribution',fontsize=12)
        dir = out_loc + 'Cluster ' + str(j) + '/'
        if not os.path.isdir(dir):
            os.makedirs(dir)
        plt.savefig(dir  + f.columns[i] + '.png')
        plt.gcf().clear()