import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import TensorBoard



def segment_map(ijv, ijv_segments, shape):
    im = np.zeros(shape) -1
    for i in range(len(ijv)):
        if not ijv_segments[i] == -1:
            im[ijv[i,0], ijv[i,1]] = ijv_segments[i]
    return(im)

def segmenter(images, segmentations, display_artifacts=False):
    artifacts, labels = [], []
    print('Segmenting artifacts...')
    for art_path in os.listdir(segmentations):

        '''
        Load segmentations
        '''
        ijv = np.load(segmentations + art_path)
        ijv_img, ijv_segments = ijv[:,:3], ijv[:,3]
        roi = segment_map(ijv_img, ijv_segments, (869,1152))

        '''
        Load original image
        '''
        img = cv2.imread(images + art_path[:-4] + '.tif')
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        '''
        Isolate the segmented artifacts
        '''
        art_labels = np.unique(ijv_segments)
        for art in art_labels[1:]:
            art_id = art_path[:-4] + '-' + str(art)
            regions, image = np.copy(roi), np.copy(img)
            regions[roi == art] = 255
            regions[roi != art] = 0
            image[regions != 255] = 0
            image = np.vstack((image, np.zeros((1024-869, image.shape[1]))))  ##Pad with bottom row of zeros


            artifacts.append(image)
            labels.append(art_id)

            if display_artifacts:
                plt.imshow(image, cmap='gray')
                plt.show()

    return np.array(artifacts), np.array(labels)






'''
Location of artifact segmentations
'''
mask_loc = '/home/miuser/Dentistry/DBSCAN/ROI-masks/'

'''
Location of original images
'''
im_loc = '/home/miuser/Dentistry/circle-removed/'

'''
Location to store encoded segmentations
'''
autoencoder_loc = '/home/miuser/Dentistry/autoencoder/'

'''
Write original segmentations?
'''
write_originals = False




'''
Create directory structure
'''
if not os.path.isdir(autoencoder_loc + 'originals/'):
    os.makedirs(autoencoder_loc + 'originals/')
if not os.path.isdir(autoencoder_loc + 'recovered/'):
    os.makedirs(autoencoder_loc + 'recovered/')
if not os.path.isdir(autoencoder_loc + 'encoded/'):
    os.makedirs(autoencoder_loc + 'encoded/')



'''
Segment the artifacts based on the DBSCAN segmentations
'''
artifacts, ids = segmenter(im_loc, mask_loc, display_artifacts=False)



input_img = Input(shape=(1024, 1152, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same', name='encoder')(x)

## Encoded

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)

decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
print(autoencoder.summary())
autoencoder.compile('adadelta', loss='binary_crossentropy')


x_train = artifacts[:1500].astype('float32') / 255.
x_test = artifacts[1500:].astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 1024, 1152, 1))
x_test = np.reshape(x_test, (len(x_test), 1024, 1152, 1))
ids_train = ids[:1500]
ids_test = ids[1500:]

autoencoder.fit(x_train, x_train,
                epochs=2,
                batch_size=8,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])


decoded_imgs_train = autoencoder.predict(x_train, batch_size=2)
decoded_imgs_test = autoencoder.predict(x_test, batch_size=2)
decoded_imgs = np.concatenate((decoded_imgs_train, decoded_imgs_test))

encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder').output)
encoded_imgs_train = encoder.predict(x_train)
encoded_imgs_test = encoder.predict(x_test)
encoded_imgs = np.concatenate((encoded_imgs_train, encoded_imgs_test))


'''
Write decoded images
'''
for i in range(len(decoded_imgs)):
    im = decoded_imgs[i,:,:,0]
    id = ids[i]
    im = im[:869]
    plt.imsave(autoencoder_loc + 'recovered/' + id + '.tif', im, cmap='gray')

'''
Write encoded images
'''
for i in range(len(encoded_imgs)):
    im = encoded_imgs[i].reshape((64, 72 * 8))
    id = ids[i]
    im = im[:869]
    plt.imsave(autoencoder_loc + 'encoded/' + id + '.tif', im, cmap='gray')



n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(1024, 1152))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(1024, 1152))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()




n = 10
plt.figure(figsize=(20, 8))
for i in range(n):
    ax = plt.subplot(1, n, i+1)
    plt.imshow(encoded_imgs[i].reshape(72, 8 * 64).T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
