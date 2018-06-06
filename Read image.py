import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import pandas as pd

def read_image(dir_name):
    list = []
    for filename in os.listdir(dir_name):
        realname = dir_name + filename
        list.append(mpimg.imread(realname))
        a = np.array(list)
    return(a)

dir_artifact = 'PSP_Plate_Artifact_Images/Sample_Artifact_Images/'
sam = read_image(dir_artifact)
# read sample artifact images
label = os.listdir(dir_artifact)
# read image labels
control = mpimg.imread('PSP_Plate_Artifact_Images/CONTROL_0.tif')
# read a new plate as the background control
list = []
for i in range(len(label)):
    list.append(np.sum(sam[i].astype('int64') - control.astype('int64')))


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

plt.scatter(use_time, list)
# plot damge degree against time of use
