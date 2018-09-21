import os
import cv2
import numpy as np

art_loc = 'kittler/'
list = []
for i in os.listdir(art_loc):
    img_loc = art_loc + i
    img = cv2.imread(img_loc, cv2.IMREAD_GRAYSCALE)
    white = np.sum(img == 255)
    print(i, white)
    if white > 960000:
        list.append(i)
print(len(list))


# kit_loc = 'kittler/'
# o_loc = 'otsu/'
# out_loc = 'kittler_otsu/'
#
# if not os.path.isdir(out_loc):
#     os.makedirs(out_loc)
#
# for i in os.listdir(o_loc):
#     loc_kit = kit_loc + i
#     loc_o = o_loc + i
#     img_kit = cv2.imread(loc_kit, cv2.IMREAD_GRAYSCALE)
#     img_o = cv2.imread(loc_o, cv2.IMREAD_GRAYSCALE)
#     stack = np.hstack((img_kit, img_o))
#     cv2.imwrite(out_loc + i, stack)