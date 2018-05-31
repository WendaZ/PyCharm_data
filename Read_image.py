import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os


dir_name = 'PSP_Plate_Artifact_Images/Sample_Artifact_Images/'
img = np.zeros(50)
list = []
for filename in os.listdir(dir_name):
    realname = dir_name + filename
    list.append(mpimg.imread(realname))
print(list)
print(len(list))

