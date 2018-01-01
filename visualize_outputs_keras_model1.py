#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPool2D
from keras.models import Sequential

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', input_shape=(64, 64, 1)))
conv1out = Activation('relu')
model.add(conv1out)
maxpool1out = MaxPool2D()
model.add(maxpool1out)
model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
conv2out = Activation('relu')
model.add(conv2out)
model.add(MaxPool2D())
model.add(Conv2D(256, kernel_size=(3, 3), padding='same'))
conv3out = Activation('relu')
model.add(conv3out)
model.add(MaxPool2D())
model.add(Conv2D(512, kernel_size=(3, 3), padding='same'))
conv4out = Activation('relu')
model.add(conv4out)
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.load_weights('test.h5')


def get_image():
  img_size = 64
  min_rect_size = 5
  max_rect_size = 50
  img = np.zeros((img_size, img_size, 1))
  num = np.random.choice(range(3))
  if num == 0:  # equal
    xsize = np.random.randint(min_rect_size, max_rect_size)
    ysize = xsize
    print('Shape: {}'.format('equal'))
  elif num == 1:  # width
    ysize = np.random.randint(max_rect_size / 2, max_rect_size)
    ratio = np.random.choice([1.5, 2, 3])
    xsize = int(ysize / ratio)
    print('Shape: {}'.format('width'))
  elif num == 2:  # long
    xsize = np.random.randint(max_rect_size / 2, max_rect_size)
    ratio = np.random.choice([1.5, 2, 3])
    ysize = int(xsize / ratio)
    print('Shape: {}'.format('long'))

  x = np.random.randint(0, img_size - xsize)
  y = np.random.randint(0, img_size - ysize)
  img[x:x + xsize, y:y + ysize, 0] = 1.
  return img


img_to_visualize = get_image()
plt.imshow(img_to_visualize[..., 0])
plt.show()

img_to_visualize = np.expand_dims(img_to_visualize, 0)


def layer_to_visualize(layer):
  inputs = [K.learning_phase()] + model.inputs

  _convout1_f = K.function(inputs, [layer.output])

  def convout1_f(X):
    # The [0] is to disable the training phase flag
    return _convout1_f([0] + [X])

  convolutions = convout1_f(img_to_visualize)
  convolutions = np.squeeze(convolutions)

  print ('Shape of conv:', convolutions.shape)

  num = convolutions.shape[2]
  n = int(np.ceil(np.sqrt(num)))

  # Visualization of each filter of the layer
  fig = plt.figure()
  for i in range(num):
    ax = fig.add_subplot(n, n, i + 1)
    ax.imshow(convolutions[..., i], cmap='gray')
  plt.show()
  fig.close()


# Specify the layer to want to visualize
layer_to_visualize(conv1out)
# layer_to_visualize(conv2out)
# layer_to_visualize(conv3out)
# layer_to_visualize(conv4out)
