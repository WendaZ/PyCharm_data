import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import pandas as pd

def read_image(dir_name):
    list = []
    i = 0
    for filename in os.listdir(dir_name):
        realname = dir_name + filename
        list.append(mpimg.imread(realname))
        i += 1
        print("Read %s images" %i)
    a = np.array(list).astype("uint8")
    return(a)

dir_artifact = 'PSP_Plate_Artifact_Images/Sample_Artifact_Images/'
sam = read_image(dir_artifact)
# read sample artifact images
label = os.listdir(dir_artifact)
# read image labels
control = mpimg.imread('PSP_Plate_Artifact_Images/CONTROL_0.tif')
# read a new plate as the background control

list = []
sam = sam.astype('int64')
control = control.astype('int64')
x = sam - control
for i in range(len(label)):
    list.append(np.sum(x[i]))
#sum all the differences to somewhat represent the degree of damage

df = pd.read_excel('Master_sheet.xlsx', sheetname = 'Sheet 1')
# read the spreadsheet

uses = df['Number of Uses']
use_time = []
for i in uses:
    use_time.append(i)
# store the number of uses for each plate

indx_unknown = []
for index, i in enumerate(use_time):
    if i == 'UNKNOWN':
        indx_unknown.append(index)
# find out about the indices of plates with unknown uses

ind = indx_unknown[::-1]
for i in ind:
    del list[i]
    del use_time[i]
# delete the corresponding items in both lists
print(list)
print(use_time)
plt.scatter(use_time, list)
# plot damge degree against time of use

def show(a):
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(sam[a], cmap='gray')
    axarr[1].imshow(x[a], cmap='gray')
# show the subtratced plot together with the original plot
ave = []
for i in range(sam.shape[0]):
    ave.append(np.average(sam[i]))
# calculate the average of the each image

for i in range(sam.shape[0]):
    sam[i] = sam[i] / ave[i]