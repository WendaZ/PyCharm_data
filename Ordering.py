import cv2
import numpy as np
import os

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


art_loc = 'Sample_Artifact_Images/'
list = []
order_loc = 'order_brightness-standardized/'
out_loc = 'ordered-otsu/'
out_loc2 = 'ordered-kittler/'
out_loc3 = 'ordered-comparison_6/'
out_loc4 = 'ordered-comparison_10/'
out_loc5 = 'segmented_artifacts/'
out = np.zeros(869*1152).reshape(869,1152)

if not os.path.isdir(out_loc5):
    os.makedirs(out_loc5)

# for i in os.listdir(art_loc):
#     img_loc = art_loc + i
#     img = cv2.imread(img_loc, cv2.IMREAD_GRAYSCALE)
#     background = np.percentile(img, 50)
#     img1 = img - background
#     img1[img1< 0] = 0
#     sum = np.sum(img1)
#     clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
#     equ = clahe.apply(img)
#     ret3, th3 = cv2.threshold(equ, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#     Kittler(equ, out)
#     stack = np.hstack((img, th3, out))
#     # cv2.imwrite(out_loc + str(sum) + "_" + i, th3)
#     # cv2.imwrite(out_loc2 + str(sum) + "_" + i, out)
#     cv2.imwrite(out_loc4 + str(sum) + "_" + i, stack)
#     print(i)

for i in os.listdir(art_loc):
    img_loc = art_loc + i
    img = cv2.imread(img_loc, cv2.IMREAD_GRAYSCALE)
    background = np.percentile(img, 50)
    img1 = img - background
    img1[img1 < 0] = 0
    sum = np.sum(img1)
    clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
    equ = clahe.apply(img)
    if sum <= 1699653:
        Kittler(equ, out)
        cv2.imwrite(out_loc5 + "_" + i, out)
    else:
        ret3, th3 = cv2.threshold(equ, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite(out_loc5 + "_" + i, th3)
    # stack = np.hstack((img, th3, out))
    # cv2.imwrite(out_loc3 + str(sum) + "_" + i, stack)
    print(i)

