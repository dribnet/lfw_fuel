from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from scipy.misc import imresize 

# this allows the example to be run in-repo
# (or can be removed if lfw_fuel is installed)
import sys
sys.path.append('.')

from lfw_fuel import lfw

'''
    Train a simple convnet on the LFW dataset.
'''

batch_size = 128
nb_epoch = 12
feature_width = 32
feature_height = 32
downsample_size = 32

def crop_and_downsample(originalX):
    """
    Starts with a 250 x 250 image.
    Crops to 128 x 128 around the center.
    Downsamples the image to (downsample_size) x (downsample_size).
    Returns an image with dimensions (channel, width, height).
    """
    current_dim = 250
    target_dim = 128
    margin = int((current_dim - target_dim)/2)
    left_margin = margin
    right_margin = current_dim - margin

    # newim is shape (6, 128, 128)
    newim = originalX[:, left_margin:right_margin, left_margin:right_margin]

    # resized are shape (feature_width, feature_height, 3)
    feature_width = feature_height = downsample_size
    resized1 = imresize(newim[0:3,:,:], (feature_width, feature_height), interp="bicubic", mode="RGB")
    resized2 = imresize(newim[3:6,:,:], (feature_width, feature_height), interp="bicubic", mode="RGB")

    # re-packge into a new X entry
    newX = np.concatenate([resized1,resized2], axis=2)

    # the next line is important.
    # if you don't normalize your data, all predictions will be 0 forever.
    newX = newX/255.0

    return newX

(X_train, y_train), (X_test, y_test) = lfw.load_data("deepfunneled")

# the data, shuffled and split between train and test sets
X_train = np.asarray([crop_and_downsample(x) for x in X_train])
X_test  = np.asarray([crop_and_downsample(x) for x in X_test])

# print shape of data while model is building
print("{1} train samples, {2} channel{0}, {3}x{4}".format("" if X_train.shape[1] == 1 else "s", *X_train.shape))
print("{1}  test samples, {2} channel{0}, {3}x{4}".format("" if X_test.shape[1] == 1 else "s", *X_test.shape))

model = Sequential()

model.add(Conv2D(32, (5,5), input_shape=(downsample_size,downsample_size,6), padding='same', data_format='channels_last', activation='relu'))
model.add(Conv2D(32, (5,5), padding='same', data_format='channels_last', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), data_format='channels_last'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', metrics=['binary_accuracy'], optimizer='adadelta')
model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
