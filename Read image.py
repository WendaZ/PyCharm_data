import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

def read_image(dir_name):
    list = []
    for filename in os.listdir(dir_name):
        realname = dir_name + filename
        list.append(mpimg.imread(realname))
        a = np.array(list)
    return(a)

dir_artifact = 'PSP_Plate_Artifact_ Images/Sample_Artifact_Images/'
sam = read_image(dir_artifact)
# read sample artifact images

dir_bends1 = 'PSP_Plate_Artifact_ Images/Control_Images/Bends#1/'
ctrl_be1 = read_image(dir_bends1)
# read bended control images

dir_bites1 = 'PSP_Plate_Artifact_ Images/Control_Images/Bitemarks#1/'
ctrl_bi1 = read_image(dir_bites1)
# read bitten control images

dir_bites2 = 'PSP_Plate_Artifact_ Images/Control_Images/Bitemarks#2/'
ctrl_bi2 = read_image(dir_bites2)
# read bitten control images set 2

