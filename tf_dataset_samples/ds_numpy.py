#!/usr/bin/python
# -*- coding: utf-8 -*-
import time

import cv2
import numpy as np
import tensorflow as tf


with open('train.csv') as f:
  lines = [line.strip().split(',') for line in f.readlines()]

label_map = {'猫': 0, '狗': 1}

images = np.ndarray((len(lines), 224, 224, 3), float)
labels = np.ndarray((len(lines)), int)

for i, line in enumerate(lines):
  image = cv2.imread(line[0])
  image = cv2.resize(image, (224, 224))
  label = label_map[line[1]]

  images[i] = image
  labels[i] = label

batch_size = 2
data = tf.data.Dataset.from_tensor_slices((images, labels))
data = data.batch(batch_size)
iterator = tf.data.Iterator.from_structure(data.output_types,
                                           data.output_shapes)
init_op = iterator.make_initializer(data)
tt = time.time()
with tf.Session() as sess:
  sess.run(init_op)
  for i in range(100):
    try:
      _, labels = iterator.get_next()
      labels = sess.run(labels)
      print('{} -> {}'.format(i, labels))
    except tf.errors.OutOfRangeError:
      sess.run(init_op)

  print(time.time() - tt)
