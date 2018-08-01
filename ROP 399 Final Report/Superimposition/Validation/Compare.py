import numpy as np
import os
from skimage.measure import compare_ssim as ssim
import matplotlib.image as mpimg
import skimage

dir1 = 'Validation/Real/'
dir2 = 'Validation/Superimposed/'

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

def compare(img1, img2, win):
    return ssim(img1, img2, win_size=win)


def cut(img,a,b):
    copy = np.copy(img)
    copy[copy > b] = 0
    copy[copy <= a] = 0
    return copy

real = read_image(dir1)
superimposed = read_image(dir2)
label = os.listdir(dir1)

a = 0
b = 0
for i in range(len(label)):
    score = compare(cut(superimposed[1],30,40),cut(real[i],30,40),3)
    print('Superimposed 2 vs Real %d :' %(i+1), score)
    if score > a:
        a = score
        b = i + 1

print('Superimposed 2 is most similar to real image', b)