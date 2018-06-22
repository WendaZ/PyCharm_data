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

dir_artifact = 'Contrast_limited/'
sam = read_image(dir_artifact)
# read sample artifact images
label = os.listdir(dir_artifact)
# read image labels
a = 0
def calc_area(a, his):
    sum = 0
    for i, s in enumerate(his):
        if i >= a:
            sum += s
    return int(sum)
b = 254
SUM = []
for i, s in enumerate(sam):
    his = cv2.calcHist([s], [0], None, [256], [0, 256])
    SUM.append(calc_area(b, his))
    path ="C:/Users/MedImage7271/Desktop/PyCharm Projects/Dental plate artifact classification/CHistogram/" + label[i]
    fig = plt.plot(range(len(his)), his)
    plt.savefig(path)
    a += 1
    plt.gcf().clear()
    print("Saved %s images" %a)

# plt.plot(range(len(SUM)), SUM)
# plt.savefig("C:/Users/MedImage7271/Desktop/PyCharm Projects/Dental plate artifact classification/Histogram/#_of_pixels_over" + str(b))
