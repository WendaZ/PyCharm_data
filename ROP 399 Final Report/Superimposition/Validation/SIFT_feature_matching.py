import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
# from cv2 import xfeatures2d

real_images_source = 'Validation/Real'
superimposed_images_source = 'Validation/Superimposed'

#
#
# for img_path in os.listdir(real_images_source):
#     img = cv2.imread(real_images_source + '/' + img_path)
#
real_img = cv2.imread(real_images_source + '/' + 'ART_00060.tif')
superimposed_image = cv2.imread(superimposed_images_source + '/' + 'ART_00060.tif')
# gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# #img_as_array = np.array(gray)
#
orb = cv2.ORB_create()
#sift = cv2.xfeatures2d.SIFT_create()
# kp = sift.detect(gray,None)
# img=cv2.drawKeypoints(gray,kp, img,  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# '''
# Apply SIFT algorithm to the image...
# '''
# kp, des = sift.detectAndCompute(gray,None)
#
# cv2.imshow("real_image_1", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

kp1, des1 = orb.detectAndCompute(real_img, None)
kp2, des2 = orb.detectAndCompute(superimposed_image, None)

'''
Next create a BFMatcher object with distance measurement cv2.NORM_HAMMING and crossCheck is switched on for better
results.
'''

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

# Match descriptors

matches = bf.match(des1, des2)

# Sort them in order of their distance

matches = sorted(matches, key = lambda x:x.distance)

# draw first 10 matches

img3 = cv2.drawMatches(real_img, kp1, superimposed_image, kp2, matches, None, flags = 2)

plt.title('REAL                                          SUPERIMPOSITION')
plt.imshow(img3), plt.show()