import pandas as pd
import numpy as np
import cv2
import os

# def segment_map(ijv, ijv_segments, shape):
#     im = np.zeros(shape)
#     for i in range(len(ijv)):
#         if not ijv_segments[i] == -1:
#             im[ijv[i,0], ijv[i,1]] = ijv_segments[i]
#     return(im)


image_loc = 'DBSCAN/Images/'
label_loc = 'DBSCAN/ROI-masks/'
cluster_loc = 'DBSCAN/KMeans-scaled/clusters.csv'
out_loc = 'C:/Users/MedImage7271/Desktop/PyCharm Projects/Dental plate artifact classification/DBSCAN/Rectangle-scaled/'
if not os.path.isdir(out_loc):
    os.makedirs(out_loc)
color_loc = 'colors.csv'

colormap = pd.read_csv(color_loc, index_col=0).values
color = [tuple(i) for i in colormap][1:]
color_used = {}
cluster = pd.read_csv(cluster_loc, header = None).values
dic = {}

for i in range(len(cluster)):
    dic[cluster[i][0]] = cluster[i][1]

count = {}

cluster_number = 5

for i in range(cluster_number):
    count['Cluster' + str(i)] = 0


for i in range(len(os.listdir(image_loc))):
    img = cv2.imread(image_loc + os.listdir(image_loc)[i])
    ijv = np.load(label_loc + os.listdir(label_loc)[i])
    ijv_img, ijv_segments = ijv[:, :3], ijv[:, 3]
    # roi = segment_map(ijv_img, ijv_segments, (869, 1152))
    artifacts = np.unique(ijv_segments)[1:]
    for j in artifacts:
        clus = dic[os.listdir(image_loc)[i][0:-4] + '-' + str(j)]
        art = ijv[ijv[:, -1] == j]
        top_left = (min(int(np.max(art[:, 1])) + 10 ,1152),max(int(np.min(art[:, 0])) - 10,0))
        bottom_right = (max(int(np.min(art[:, 1])) -10, 0),min(int(np.max(art[:, 0])) + 10, 869))
        img = cv2.rectangle(img, top_left, bottom_right, [int(n) for n in color[clus]][::-1], 2)
        if color[clus] not in color_used.values():
            color_used['Cluster' + str(clus)] = color[clus]
        count['Cluster' + str(clus)] += 1
    cv2.imwrite(out_loc + os.listdir(image_loc)[i], img)
    print(os.listdir(image_loc)[i])

print(color_used)
print(count)