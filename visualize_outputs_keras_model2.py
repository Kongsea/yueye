#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import keras
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np


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


model_path = 'test.h5'
model = keras.models.load_model(model_path)
print('Using {}'.format(model_path))


def get_layer_outputs():
  img_to_visualize = get_image()
  plt.imshow(img_to_visualize[..., 0])
  plt.show()
  img_to_visualize = np.expand_dims(img_to_visualize, 0)
  outputs = [layer.output for layer in model.layers]  # all layer outputs
  comp_graph = [K.function([model.input] + [K.learning_phase()], [output])
                for output in outputs]  # evaluation functions

  # Testing
  layer_outputs_list = [op([img_to_visualize, 1.]) for op in comp_graph]
  layer_outputs = []

  for layer_output in layer_outputs_list:
    print(layer_output[0][0].shape, end='\n-------------------\n')
    layer_outputs.append(layer_output[0][0])

  return layer_outputs


def plot_layer_outputs(layer_number):
  layer_outputs = get_layer_outputs()

  x_max = layer_outputs[layer_number].shape[0]
  y_max = layer_outputs[layer_number].shape[1]
  n = layer_outputs[layer_number].shape[2]

  L = []
  for i in range(n):
    L.append(np.zeros((x_max, y_max)))

  for i in range(n):
    for x in range(x_max):
      for y in range(y_max):
        L[i][x][y] = layer_outputs[layer_number][x][y][i]

  fig = plt.figure()
  for i, c in enumerate(L):
    ax = fig.add_subplot(np.ceil(n**0.5), np.ceil(n**0.5), i + 1)
    ax.imshow(c, cmap='gray')
  plt.show()


plot_layer_outputs(3)
