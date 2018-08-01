from Read_image import read_image
import numpy as np
import os
import matplotlib.image as mpimg
import skimage
import matplotlib.pyplot as plt
import cv2
import math
import time
from cv2_superposition import cut

def show(a):
    plt.imshow(a,cmap = 'gray')



# def clipped_add(a,b):
#     a = a.astype('uint16')
#     b = b.astype('uint16')
#     c = a+b
#     np.clip(c, 0, 255)
#     c = c.astype('uint8')
#     return c

def exp_w(a, b, x):
    e = math.e
    y = a*(b**x)
    return y

def get_para(x1,y1,x2,y2):
    b = (y2/y1) ** (1/(x2-x1))
    a = y1/(b ** x1)
    return a, b

dir1 = 'selected_blank/' #original images of the 25 selected artifacts
dir2 = 'Test/' #contrast limited images of the 25 selected artifacts
dir3 = 'PSP_Plate_Artifact_Images/Phantom Images/'
label = os.listdir(dir1)

blank = read_image(dir1)
cl = read_image(dir2)
real = read_image(dir3)
cmos = cv2.resize(cv2.cvtColor(skimage.img_as_ubyte(mpimg.imread('CMOS Control Image.PNG')), cv2.COLOR_BGR2GRAY), blank.shape[1:][::-1])

a = 0.15905414575341018
b = 1.0057432466391787

a, b = get_para(30, 0.1, 255, 0.5)

for i in range(25):
    w = exp_w(a, b, cl[i])
    m = cl[i] * w
    cmos = cmos.astype('float64')
    blend = cv2.addWeighted(m, 1, cmos, 0.85, 0)
    high = cut(blank[i],110,255).astype('float64')
    high[high > 0] += 80
    high[high > 255] = 255
    blend = np.maximum(blend, high)
    blend[blank[1] == 0] = 0
    cv2.imwrite('C:/Users/MedImage7271/Desktop/PyCharm Projects/Dental plate artifact classification/Superimposed_exp/cl/' + label[i], blend)
    print('Saved %d images-----equalized' % (i + 1))

print("a: ", a)
print("b: ", b)

# a = 2.185095253170872
# b = 0.9935584113333836
# a, b = get_para(30, 2.2, 150, 0.7)
# for i in range(25):
#     w = exp_w(a, b, blank[i])
#     m = blank[i] * w
#     cmos = cmos.astype('float64')
#     blend = cv2.addWeighted(m, 1, cmos, 0.8, -20)
#     blend[blank[1] == 0] = 0
#     cv2.imwrite('C:/Users/MedImage7271/Desktop/PyCharm Projects/Dental plate artifact classification/Superimposed_exp/ori/' + label[i], blend)
#     print('Saved %d images-----original' % (i + 1))
#
#     # his = cv2.calcHist(blank[i], [0], None, [256], [0, 256]).astype('uint8')
#     # list = []
#     # for i in his:
#     #     list.append(i[0])
#     # z = np.linspace(0, 255, 3001)
#     #
#     # fig, ax1 = plt.subplots()
#     # ax1.plot(z, exp_w(a, b, z), 'b-')
#     # ax2 = ax1.twinx()
#     # ax2.plot(range(256), list, 'r')
#     # fig.tight_layout()
#     # plt.show(block=False)
#     # plt.pause(0.5)
#     # plt.close()
#
# print("a: ", a)
# print("b: ", b)