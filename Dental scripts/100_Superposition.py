import numpy as np
import os
import cv2
import math

def exp_w(a, b, x):
    e = math.e
    y = a*(b**x)
    return y

def get_para(x1,y1,x2,y2):
    b = (y2/y1) ** (1/(x2-x1))
    a = y1/(b ** x1)
    return a, b

def cut(img,a,b):
    copy = np.copy(img)
    copy[copy > b] = 0
    copy[copy <= a] = 0
    return copy

dir1 = 'Selected Periapical Images/'
dir2 = 'Test/'
dir3 = 'selected_blank/'

label_1 = os.listdir(dir1)
label_2 = os.listdir(dir2)

a, b = get_para(30, 0.1, 255, 0.5)

for i in range(len(label_1)):
    filepath = dir1 + label_1[i]
    teeth = cv2.resize(cv2.imread(filepath, cv2.IMREAD_UNCHANGED),(1152,869)).astype('float64')
    for j in range(len(label_2)):
        filename = dir2 + label_2[j]
        art = cv2.cvtColor(cv2.imread(filename, cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2RGBA).astype('float64')
        filenamee = dir3 + label_2[j]
        blank = cv2.cvtColor(cv2.imread(filenamee, cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2RGBA).astype('float64')
        w = exp_w(a, b, art)
        m = art * w
        blend = cv2.addWeighted(m, 1, teeth, 0.85, 0)
        high = cut(blank, 110, 255)
        high[high > 0] += 80.
        high[high > 255] = 255.
        blend = np.maximum(blend, high)
        blend[blank == 0] = 0
        cv2.imwrite('C:/Users/MedImage7271/Desktop/PyCharm Projects/Dental plate artifact classification/Final_superimposed/' +
            label_1[i][0:-4] + '_' + label_2[j], blend)
        print('Saved %d images' %(i * len(label_2) + (j+1)))