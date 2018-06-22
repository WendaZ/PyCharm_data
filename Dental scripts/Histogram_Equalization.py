import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import cv2
import skimage

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

dir_artifact = 'PSP_Plate_Artifact_Images/Sample_Artifact_Images/'
sam = read_image(dir_artifact)
label = os.listdir(dir_artifact)

for i, s in enumerate(sam):
    clahe = cv2.createCLAHE(clipLimit= 10.0, tileGridSize=(8,8))
    equ = clahe.apply(s)
    path = "C:/Users/MedImage7271/Desktop/PyCharm Projects/Dental plate artifact classification/Contrast_limited/" + label[i]
    cv2.imwrite(path, equ)
    print("Saved %s images" %(i+1))