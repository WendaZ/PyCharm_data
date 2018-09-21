import os
import numpy as np
np.set_printoptions(threshold=np.nan)
import cv2
import matplotlib.pyplot as plt


equ_loc = 'Contrast_limited/'
ori_loc = 'Sample_Artifact_Images/'

out_loc1 = 'otsu_stacked/'
out_loc2 = 'otsu/'
out_loc3 = 'kittler_stacked/'
out_loc4 = 'kittler/'

if not os.path.isdir(out_loc1):
    os.makedirs(out_loc1)
if not os.path.isdir(out_loc2):
    os.makedirs(out_loc2)
if not os.path.isdir(out_loc3):
    os.makedirs(out_loc3)
if not os.path.isdir(out_loc4):
    os.makedirs(out_loc4)
'''
Otsu method
'''
a = cv2.imread('Sample_Artifact_Images/ART_00014.tif', cv2.IMREAD_GRAYSCALE)
for i in os.listdir(equ_loc):
    img_loc = equ_loc + i
    img_loc2 = ori_loc + i
    img = cv2.imread(img_loc, cv2.IMREAD_GRAYSCALE)
    img_ori = cv2.imread(img_loc2, cv2.IMREAD_GRAYSCALE)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret3, th3 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    stack = np.hstack((img_ori, th3))
    cv2.imwrite(out_loc1 + i, stack)
    cv2.imwrite(out_loc2 + i, th3)
    print(i)
exit(0)
'''
Kittler Method
'''
out = np.zeros(869*1152).reshape(869,1152)
def Kittler(im, out):
    """
    The reimplementation of Kittler-Illingworth Thresholding algorithm by Bob Pepin
    Works on 8-bit images only
    Original Matlab code: https://www.mathworks.com/matlabcentral/fileexchange/45685-kittler-illingworth-thresholding
    Paper: Kittler, J. & Illingworth, J. Minimum error thresholding. Pattern Recognit. 19, 41–47 (1986).
    """
    h,g = np.histogram(im.ravel(),256,[0,256])
    h = h.astype(np.float)
    g = g.astype(np.float)
    g = g[:-1]
    c = np.cumsum(h)
    m = np.cumsum(h * g)
    s = np.cumsum(h * g**2)
    sigma_f = np.sqrt(s/c - (m/c)**2)
    cb = c[-1] - c
    mb = m[-1] - m
    sb = s[-1] - s
    sigma_b = np.sqrt(sb/cb - (mb/cb)**2)
    p =  c / c[-1]
    v = p * np.log(sigma_f) + (1-p)*np.log(sigma_b) - p*np.log(p) - (1-p)*np.log(1-p)
    v[~np.isfinite(v)] = np.inf
    idx = np.argmin(v)
    t = g[idx]
    out[:,:] = 0
    out[im >= t] = 255


for i in os.listdir(equ_loc):
    img_loc = equ_loc + i
    img = cv2.imread(img_loc, cv2.IMREAD_GRAYSCALE)
    Kittler(img, out)
    stack = np.hstack((img, out))
    cv2.imwrite(out_loc3 + i, stack)
    cv2.imwrite(out_loc4 + i, out)
    print(i)