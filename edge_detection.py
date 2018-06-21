import os
import cv2
import numpy as np


'''
Location of artifact images
'''
imsource = "C:/Users/MedImage7271/Desktop/PyCharm Projects/Dental plate artifact classification/PSP_Plate_Artifact_Images/Sample_Artifact_Images/"

'''
Location to store filtered images
'''
filtered_loc = 'C:/Users/MedImage7271/Desktop/PyCharm Projects/Dental plate artifact classification/Canny_filtered/'

'''
Display the images? Hit any key to view next one...
'''
display_ims = False

'''
Implement the Canny algorithm with min value 10, max value 45
'''
filtered = []
for img_path in os.listdir(imsource):
    print(img_path)
    print(imsource + img_path)
    img = cv2.imread(imsource + img_path)
    edges = cv2.Canny(img,10,45,apertureSize = 3) ## 10, 45
    filtered.append((img_path, edges))

    ## View the image
    if display_ims:
        cv2.imshow("image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

'''
Write the filtered images
'''
if not os.path.isdir(filtered_loc):
    os.makedirs(filtered_loc)

for im in filtered:
    cv2.imwrite(filtered_loc + im[0], im[1])