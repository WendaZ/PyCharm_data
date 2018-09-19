import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


art_loc = 'artifacts/'
out_loc = 'otsu/'

if not os.path.isdir(out_loc):
    os.makedirs(out_loc)

'''
Otsu method
'''

for i in os.listdir(art_loc):
    img_loc = art_loc + i
    img = cv2.imread(img_loc, cv2.IMREAD_GRAYSCALE)
    # plt.imshow(img, cmap='gray')
    # plt.show()
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(out_loc + i, th3)

'''
Kittler Method
'''

# def Kittler(im, out):
#     """
#     The reimplementation of Kittler-Illingworth Thresholding algorithm by Bob Pepin
#     Works on 8-bit images only
#     Original Matlab code: https://www.mathworks.com/matlabcentral/fileexchange/45685-kittler-illingworth-thresholding
#     Paper: Kittler, J. & Illingworth, J. Minimum error thresholding. Pattern Recognit. 19, 41–47 (1986).
#     """
#     h,g = np.histogram(im.ravel(),256,[0,256])
#     h = h.astype(np.float)
#     g = g.astype(np.float)
#     g = g[:-1]
#     c = np.cumsum(h)
#     m = np.cumsum(h * g)
#     s = np.cumsum(h * g**2)
#     sigma_f = np.sqrt(s/c - (m/c)**2)
#     cb = c[-1] - c
#     mb = m[-1] - m
#     sb = s[-1] - s
#     sigma_b = np.sqrt(sb/cb - (mb/cb)**2)
#     p =  c / c[-1]
#     v = p * np.log(sigma_f) + (1-p)*np.log(sigma_b) - p*np.log(p) - (1-p)*np.log(1-p)
#     v[~np.isfinite(v)] = np.inf
#     idx = np.argmin(v)
#     t = g[idx]
#     out[:,:] = 0
#     out[im >= t] = 255
