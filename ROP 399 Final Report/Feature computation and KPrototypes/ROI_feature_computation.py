import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import itemfreq
import os
import cv2
import imutils

def segment_map(ijv, ijv_segments, shape):
    im = np.zeros(shape)
    for i in range(len(ijv)):
        if not ijv_segments[i] == -1:
            im[ijv[i,0], ijv[i,1]] = ijv[i,2]
    return(im)




'''
Location of artifact images
'''
imsource = "circle-removed/"

'''
Location of artifact segmentations
'''
mask_loc = 'DBSCAN/ROI-masks/'

'''
Location to store features
'''
features_loc = 'C:/Users/MedImage7271/Desktop/PyCharm Projects/Dental plate artifact classification/DBSCAN/'


artifact_ids = []
NumPixels, DeltaX, DeltaY, Volume = [], [], [], []
MinIntensity, MaxIntensity, MeanIntensity, MedianIntensity = [], [], [], []
centroid, centroidX, centroidY = [], [], []
Hu1,Hu2,Hu3,Hu4,Hu5,Hu6,Hu0 = [],[],[],[],[],[],[]
shape = []

for art_path in os.listdir(mask_loc):
    print(mask_loc + art_path)

    '''
    Load segmentations
    '''
    ijv = np.load(mask_loc + art_path)
    ijv_img, ijv_segments = ijv[:,:3], ijv[:,3]


    '''
    Retreive unique artifact labels
    '''
    artifacts = np.unique(ijv_segments)[1:]

    for art in artifacts:
        artifact_ids.append(art_path[:-4] + '-' + str(art))

        '''
        Isolate single artifact
        '''
        art_ijv_img = ijv_img[ijv_segments == art]
        art_ijv_segments = ijv_segments[ijv_segments == art]

        '''
        create a segment map
        '''
        roi = np.zeros((869,1152), dtype='uint8')
        for i in range(len(art_ijv_img)):
            # roi[art_ijv_img[i][0],art_ijv_img[i][1]] = art_ijv_img[i][2]
            roi[art_ijv_img[i][0], art_ijv_img[i][1]] = 255
        '''
        Compute Number of Pixels
        '''
        NumPixels.append(len(art_ijv_segments))

        '''
        Compute Intensity properties
        '''
        MinIntensity.append(np.min(art_ijv_img[:,2]))
        MaxIntensity.append(np.max(art_ijv_img[:,2]))
        MeanIntensity.append(np.mean(art_ijv_img[:,2]))
        MedianIntensity.append(np.median(art_ijv_img[:,2]))

        '''
        Compute maximum x-y offsets
        '''
        MinX, MaxX = np.min(art_ijv_img[:,1]), np.max(art_ijv_img[:,1])
        MinY, MaxY = np.min(art_ijv_img[:,0]), np.max(art_ijv_img[:,0])
        DeltaX.append(MaxX - MinX)
        DeltaY.append(MaxY - MinY)

        '''
        Compute volume
        '''
        Volume.append(np.sum(art_ijv_img[:,2]))

        '''
         Compute the controid
        '''
        # centroid.append(tuple((np.mean(art_ijv_img[:,1]),np.mean(art_ijv_img[:,0]))))
        centroidX.append(abs(np.mean(art_ijv_img[:,1])-576))
        centroidY.append(abs(np.mean(art_ijv_img[:,0])-434.5))

        # '''
        # Compute Hu moments
        # '''
        # Hu = cv2.HuMoments(cv2.moments(roi)).flatten()
        # Hu0.append(Hu[0])
        # Hu1.append(Hu[1])
        # Hu2.append(Hu[2])
        # Hu3.append(Hu[3])
        # Hu4.append(Hu[4])
        # Hu5.append(Hu[5])
        # Hu6.append(Hu[6])

        '''
        Compute and countour and shape
        '''
        img, cnts, hir = cv2.findContours(roi, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) > 1:
            cnts = cnts[1]
        else:
            cnts = cnts[0]
        peri = cv2.arcLength(cnts, True)
        approx = cv2.approxPolyDP(cnts, 0.04 * peri, True)
        if len(approx) == 2:
            shape.append(0) #line
        elif len(approx) == 3:
            shape.append(1)   #triangle
        elif len(approx) == 4:
            shape.append(2)   #rectangle
        else:
            shape.append(3)    #circle
    # pd.DataFrame(roi).to_csv('/home/miuser/Dentistry/' + art_path[:-4] + '.csv')



artifacts = pd.DataFrame({
                'NumPixels': NumPixels,
                 'Volume': Volume,
                 'DeltaX': DeltaX,
                 'DeltaY': DeltaY,
                 'CentroidX': centroidX,
                 'CentroidY': centroidY,
                 # 'Hu0': Hu0,
                 # 'Hu1': Hu1,
                 # 'Hu2': Hu2,
                 # 'Hu3': Hu3,
                 # 'Hu4': Hu4,
                 # 'Hu5': Hu5,
                 # 'Hu6': Hu6,
                 # 'MinIntensity': MinIntensity,
                 'MaxIntensity': MaxIntensity,
                 'MedianIntensity': MedianIntensity,
                 # 'MeanIntensity': MeanIntensity,
                'Shape': shape
},
                 index=artifact_ids)
artifacts.to_csv(features_loc + 'Newfeatures.csv')