import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from skimage import data
from skimage.feature import match_template




'''
Location of artifact images
'''
imsource = "PSP_Plate_Artifact_Images/Sample_Artifact_Images/"

'''
Location to store corrected images
'''
out_loc = "circle-removed/"

show = False


if not os.path.isdir(out_loc):
    os.makedirs(out_loc)


for img_path in os.listdir(imsource):
    print(imsource + img_path)


    # if not img_path == "ART_00243.tif":
    #     continue

    '''
    Load artifact image
    '''
    img = cv2.imread(imsource + img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


    image = img
    coin = cv2.imread("C.tif")
    coin = cv2.cvtColor(coin,cv2.COLOR_BGR2GRAY)


    result = match_template(image, coin)
    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]


    fig = plt.figure(figsize=(8, 3))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3, sharex=ax2, sharey=ax2)

    ax1.imshow(coin, cmap=plt.cm.gray)
    ax1.set_axis_off()
    ax1.set_title('template')

    ax2.imshow(image, cmap=plt.cm.gray)
    ax2.set_axis_off()
    ax2.set_title('image')
    # highlight matched region
    hcoin, wcoin = coin.shape
    rect = plt.Rectangle((x, y), wcoin, hcoin, edgecolor='r', facecolor='none')
    ax2.add_patch(rect)

    ax3.imshow(result)
    ax3.set_axis_off()
    ax3.set_title('`match_template`\nresult')
    # highlight matched region
    ax3.autoscale(False)
    ax3.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)

    # plt.show()

    print(y)
    print(y+hcoin)
    print(x)
    print(x+wcoin)

    pd.DataFrame(result[y:y+hcoin,x:x+wcoin]).to_csv('test.csv')

    hist,bins = np.histogram(img.flatten(),256,[0,256])
    print(hist)
    hist[hist < 5000] = 0
    print(np.argmax(hist))
    print(hist[np.argmax(hist)])
    img[y:y+hcoin,x:x+wcoin][img[y:y+hcoin,x:x+wcoin] > 45] = 20
    cv2.imwrite(out_loc + img_path, img)