import os
import cv2
import copy
import numpy as np
import colorsys
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import coo_matrix
from scipy.stats import itemfreq
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel



def HSVToRGB(h, s, v):
    (r, g, b) = colorsys.hsv_to_rgb(h, s, v)
    return (int(255*r), int(255*g), int(255*b))

def getDistinctColors(n):
    huePartition = 1.0 / (n + 1)
    return (HSVToRGB(huePartition * value, 1.0, 1.0) for value in range(0, n))

def triplet_to_image(ijv, labels, colours, original):
    im = cv2.cvtColor(original,cv2.COLOR_GRAY2BGR)

    for pixel_num in range(len(ijv)):
        triplet = ijv[pixel_num]
        pixel_label = labels[pixel_num]
        im[triplet[0], triplet[1]] = list(colours[pixel_label+1])
        if (im[triplet[0], triplet[1]] == [0,0,0]).all():
            im[triplet[0], triplet[1]] = [triplet[2], triplet[2], triplet[2]]
    return im.astype('uint8')






'''
Location of artifact images
'''
imsource = "circle-removed/"

'''
Location to store filtered images
'''
output_loc = 'DBSCAN/'

'''
Location of colours to use
'''
color_loc = 'colors.csv'

'''
Max distance for DBSCAN (40 tends to work)
'''
DB_eps = 20

'''
Min number of samples for cluster in DBSCAN (10 tends to work)
'''
DB_min_samples = 20

'''
Display the segmented images?
'''
show_im = False



'''
Create directory structure
'''
if not os.path.isdir(output_loc):
    os.makedirs(output_loc + 'ROI-masks')

colourMapping = pd.read_csv(color_loc, index_col=0)

for img_path in os.listdir(imsource):
    print(imsource + img_path)


    # if not img_path == "ART_00002.tif":
    #     continue

    '''
    Load artifact image
    '''
    img = cv2.imread(imsource + img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    '''
    Parse as triplet format, potentially sparse
    '''
    thresh = copy.copy(img)
    thresh[thresh < 65] = 0
    sparse = coo_matrix(thresh)
    ijv = np.vstack([sparse.row, sparse.col, sparse.data]).transpose()
    # print(ijv.shape)

    '''
    Segment with DBSCAN
    '''
    db = DBSCAN(eps=DB_eps, min_samples=DB_min_samples, metric='l2', p=None, n_jobs=-1).fit(ijv)
    # print(str(len(np.unique(db.labels_))-1) + " clusters determined.")
    # print(str(len(db.components_)) + " core points determined")
    # print(db.components_.shape)

    '''
    Use colors from spreadsheet as first 12 colors, then
        generate n visually distinct colours for remaining clusters
    '''
    colourList = [tuple(i) for i in colourMapping.values]
    col_gen = getDistinctColors(len(np.unique(db.labels_)))
    residual_colours = [next(col_gen) for k in range(len(np.unique(db.labels_)))]
    colourList.extend(residual_colours)
    colourList = colourList[:len(np.unique(db.labels_))+1]
    # print(colourList)

    '''
    Label the image with the colour-coded clusters
    '''
    original_ijv = copy.copy(ijv)
    original_ijv = original_ijv.astype('uint16')

    im = triplet_to_image(original_ijv, db.labels_, colourList, img)

    # print(itemfreq(db.labels_))
    # for l in np.unique(db.labels_):
    #     print(original_ijv[db.labels_ == l][:5])

    '''
    Save the ROI labels
    '''
    img_with_mask = np.hstack((original_ijv, db.labels_.reshape(len(db.labels_),1)))
    np.save(output_loc + "ROI-masks/" + img_path[:-4] + ".npy", img_with_mask)


    '''
    Save the image
    '''
    plt.imsave(output_loc + 'Images/' + img_path, im)


    '''
    Display the image
    '''
    if show_im:
        plt.imshow(im)
        plt.show()


        fig = plt.figure()
        ax = Axes3D(fig)

        x, y, z = [], [], []
        for p in ijv:
            if p[2] > 0:
                x.append(p[0])
                y.append(p[1])
                z.append(p[2])
        ax.scatter(x, y, z)

        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = DB_eps*np.cos(u)*np.sin(v) + 0.1
        y = DB_eps*np.sin(u)*np.sin(v) + 0.05
        z = DB_eps*np.cos(v)
        ax.plot_wireframe(x, y, z, color="r")

        plt.show()