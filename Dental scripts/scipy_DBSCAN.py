import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.cluster import DBSCAN
from Read_image import read_image

dir_artifact = 'original_percentile_thresholded/'
sam = read_image(dir_artifact)
label_1 = os.listdir(dir_artifact)

for i in sam:
    sparse = coo_matrix(i)
    ijv = np.vstack([sparse.row, sparse.col, sparse.data]).transpose()
    db = DBSCAN(eps=5, min_samples=50, metric='euclidean').fit(ijv)
    print(np.unique(db.labels_))

    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(np.unique(db.labels_)))]
    break
    n_clusters_ = len(np.unique(db.labels_))
    unique_labels = set(db.labels_)

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (db.labels_ == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
