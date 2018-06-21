import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

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
label = os.listdir(dir_artifact)
sam = read_image(dir_artifact)
'''
Load bends control images as dictionary mapping id to image
	Cast as numpy array in order of increasing image id
'''
background_percentile = 20
stdized_im_loc = 'C:/Users/MedImage7271/Desktop/PyCharm Projects/Dental plate artifact classification/brightness_adjusted/'

'''
Stadardize images to [0,1] by dividing by 25535 (= 256^2 - 1)
'''
s = np.array([i/255 for i in sam]).astype('float64')


'''
Obtain a background level, representing the specified percentile
'''
background_level = np.array([np.percentile(i, background_percentile) for i in s])


'''
Scale the images, and set values outside of [0,1] to their limits
'''
scaled = np.array([s[i] - background_level[i] for i in range(len(s))])
scaled[scaled < 0] = 0
scaled[scaled > 1] = 1


'''
Revert images to 16-bit greyscale
'''
scaled_im = (scaled * 255).astype('uint16')


'''
Create directory if does not exist, overwriting files already there
'''
if not os.path.exists(stdized_im_loc):
	os.makedirs(stdized_im_loc)
os.chdir(stdized_im_loc)

for imnum in range(len(label)):
	filename = str(label[imnum]) + '.tif'

	if os.path.isfile(filename):
		os.remove(filename)

	plt.imsave(filename, scaled_im[imnum], format='tif', cmap='gray')
    print("Saved %s images" %(imnum+1))