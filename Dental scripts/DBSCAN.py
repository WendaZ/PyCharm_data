import matplotlib.image as mpimg
import numpy as np
import os
import cv2
import skimage
import matplotlib.pyplot as plt

def read_image(dir_name):
    list = []
    i = 0
    for filename in os.listdir(dir_name):
        realname = dir_name + filename
        list.append(mpimg.imread(realname))
        i += 1
        print("Read %s images" %i)
    a = np.array(list)
    a = skimage.img_as_ubyte(a)
    return(a)

dir_artifact = 'Thresholded/'
sam = read_image(dir_artifact)
label_1 = os.listdir(dir_artifact)
w_color = 0.5
w_space = 0.5
eps = 5
minP = 50

def generate_pixels(img):
    pixel = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel.append([i,j,img[i][j]])
    return pixel
def dis(a, b):
    d = w_color * (a[2] - b[2]) + w_space * ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5
    return d

def range_query(img, a, eps):
    neighbors = []
    for i in img:
        if dis(a, i) <= eps:
            neighbors.append(i)
    return neighbors
#Unclassified -1
#Noise -2
#Core point 1
#Non core point 0

def DBSCAN(img, minP, eps):
    C = 0
    label = ['Undefined'] * 1001088
    for i, n in enumerate(pixel):
        if label[i] is not 'Undefined':
            Neighbours = range_query(n, eps)
        if len(Neighbours) < minP:
            label[i] = "Noise"
        else:
            C += 1
            label[i]

