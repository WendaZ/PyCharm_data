from Read_image import read_image
import numpy as np
import os
import matplotlib.image as mpimg
import skimage
import matplotlib.pyplot as plt
import cv2

dir1 = 'selected_blank/' #original images of the 25 selected artifacts
dir2 = 'Test/' #contrast limited images of the 25 selected artifacts

blank = read_image(dir1) #original images
cl = read_image(dir2)   #contrast limited images
label = os.listdir(dir1)
cmos = cv2.resize(cv2.cvtColor(skimage.img_as_ubyte(mpimg.imread('CMOS Control Image.PNG')), cv2.COLOR_BGR2GRAY), blank.shape[1:][::-1])
#cmos image modified to 8 bit, grey scale and a shape of (869,1152)

def cut(img,a,b):
    copy = np.copy(img)
    copy[copy > b] = 0
    copy[copy <= a] = 0
    return copy
#keep pixels with a value between (a,b], setting the rest to 0

def split(img,a,b,c):
    low = cut(img, 0, a)
    mid1 = cut(img, a, b)
    mid2 = cut(img,b,c)
    high = cut(img, c, 255)
    return low, mid1, mid2, high
#split the image to pixel values (0,a]; (a,b]; (b,c]; (c,255]

def blendd(low, mid1,mid2, high, a, b, c, d):
    blend = cv2.addWeighted(cv2.addWeighted(cv2.addWeighted(cv2.addWeighted(low, a, cmos, 0.7, 0),1, mid1, b, 0), 1, mid2, c,0),1,high,d, 0)
    return blend
#blend different parts of the images together with different weights for each part

for i in range(25):
    low, mid1 ,mid2, high = split(blank[i], 30, 70, 120)
    blend = blendd(low, mid1, mid2, high, 1.2, 1, 1, 1)
    blend[blank[1] == 0] = 0 #set the value of pixels outside of the plate boundary to zero
    cv2.imwrite('C:/Users/MedImage7271/Desktop/PyCharm Projects/Dental plate artifact classification/Superimposed_ori/test/' + label[i], blend)
    print('Saved %d images' %(i+1))

    # for use of original artifacts, parameters that work the best so far on the set are:
    #   thresholds and weights: 0-30, w = 1.5
    #                           30-70, w = 1.2
    #                           70-120, w = 1.2
    #                           120-255, w = 1.0
    #   weight for cmos image: w = 0.8
    #   gamma: -10
