import numpy as np
import pandas as pd
import os
import cv2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale



def segment_map(ijv, ijv_segments, shape):
    im = np.zeros(shape) -1
    for i in range(len(ijv)):
        if not ijv_segments[i] == -1:
            im[ijv[i,0], ijv[i,1]] = ijv_segments[i]
    return(im)

def segmenter(images, segmentations, display_artifacts=False):
    artifacts, labels = [], []
    print('Segmenting artifacts...')
    for art_path in os.listdir(segmentations):

        '''
        Load segmentations
        '''
        ijv = np.load(segmentations + art_path)
        ijv_img, ijv_segments = ijv[:,:3], ijv[:,3]
        roi = segment_map(ijv_img, ijv_segments, (869,1152))

        '''
        Load original image
        '''
        img = cv2.imread(images + art_path[:-4] + '.tif')
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        '''
        Isolate the segmented artifacts
        '''
        art_labels = np.unique(ijv_segments)
        for art in art_labels[1:]:
            art_id = art_path[:-4] + '-' + str(art)
            regions, image = np.copy(roi), np.copy(img)
            regions[roi == art] = 255
            regions[roi != art] = 0
            image[regions != 255] = 0
            # image = np.vstack((image, np.zeros((1024-869, image.shape[1]))))  ##Pad with bottom row of zeros


            artifacts.append(image)
            labels.append(art_id)

            if display_artifacts:
                plt.imshow(image, cmap='gray')
                plt.show()

    return np.array(artifacts), np.array(labels)




'''
File locations
'''
mask_loc = '/home/miuser/Dentistry/DBSCAN/ROI-masks/' ## Artifact segmentations
im_loc = '/home/miuser/Dentistry/circle-removed/' ## Original images
output_loc = '/home/miuser/Dentistry/DBSCAN/segmented-originals/segmentations.csv'
PCA_output = '/home/miuser/Dentistry/PCA/scaled/'


'''
Segment the artifacts based on the DBSCAN segmentations
'''
artifacts, ids = segmenter(im_loc, mask_loc, display_artifacts=False)
artifacts = artifacts.reshape((len(artifacts), artifacts.shape[1] * artifacts.shape[2]))

'''
Number of components
'''
num_components = 75


'''
Perform the PCA
'''
print("PCA...")
pca = PCA(n_components=num_components)
features = pca.fit_transform(scale(artifacts))
PC = pca.components_
explained_var = pca.explained_variance_
explained_var_ratio = pca.explained_variance_ratio_
cumulative_var_ratio = np.cumsum(explained_var_ratio)
eigenvalues = pca.singular_values_
means = pca.mean_

pd.DataFrame(features, index=ids).to_csv(PCA_output + 'artifact_features.csv')
pd.DataFrame(PC).to_csv(PCA_output + 'principal_components.csv')
pd.DataFrame(explained_var).to_csv(PCA_output + 'explained_variance.csv')
pd.DataFrame(explained_var_ratio).to_csv(PCA_output + 'explained_variance_ratio.csv')
pd.DataFrame(cumulative_var_ratio).to_csv(PCA_output + 'cumulative_variance_ratio.csv')
pd.DataFrame(eigenvalues).to_csv(PCA_output + 'eigenvalues.csv')
pd.DataFrame(means).to_csv(PCA_output + 'means.csv')
