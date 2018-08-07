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

dir_artifact = 'selected_blank/'
sam = read_image(dir_artifact)
label = os.listdir(dir_artifact)

for i, s in enumerate(sam):
    clahe = cv2.createCLAHE(clipLimit= 4.0, tileGridSize=(64,64))
    equ = clahe.apply(s)
    # equ[equ < 10] = 0
    path = "C:/Users/MedImage7271/Desktop/PyCharm Projects/Dental plate artifact classification/Test/" + label[i]
    cv2.imwrite(path, equ)
    print("Saved %s images" %(i+1))
# s = sam[1]
# a = np.arange(0, 205, 5)
# clahe = cv2.createCLAHE(clipLimit= 10.0, tileGridSize=(8,8))
# equ = clahe.apply(s)
# for n, i in enumerate(a):
#     equ[equ<=i] = 0
#     path = "C:/Users/MedImage7271/Desktop/PyCharm Projects/Dental plate artifact classification/Thresholded/Threshold " + str(i) + ".tif"
#     cv2.imwrite(path, equ)
#     print("Saved %s images" % (n + 1))