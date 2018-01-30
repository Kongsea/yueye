#!/usr/bin/python
# -*- coding: utf-8 -*-
import time

import tensorflow as tf

label_map = {'猫': 0, '狗': 1}

with open('train.csv') as f:
  lines = [line.strip().split(',') for line in f.readlines()]


def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string, channels=3)          # (1)
  image = tf.cast(image_decoded, tf.float32)

  image = tf.image.resize_images(image, [224, 224])  # (2)
  return image, filename, label


def training_preprocess(image, filename, label):
  flip_image = tf.image.random_flip_left_right(image)                # (4)

  return flip_image, filename, label


images = []
labels = []
for line in lines:
  images.append(line[0])
  labels.append(label_map[line[1]])

images = tf.constant(images)
labels = tf.constant(labels)
images = tf.random_shuffle(images, seed=0)
labels = tf.random_shuffle(labels, seed=0)
data = tf.data.Dataset.from_tensor_slices((images, labels))

data = data.map(_parse_function, num_parallel_calls=4)
data = data.prefetch(buffer_size=2 * 10)
batched_data = data.batch(2)

iterator = tf.data.Iterator.from_structure(batched_data.output_types,
                                           batched_data.output_shapes)

init_op = iterator.make_initializer(batched_data)
tt = time.time()
with tf.Session() as sess:
  sess.run(init_op)
  for i in range(100):
    try:
      images, filenames, labels = iterator.get_next()
      print('{} -> {}'.format(i, sess.run(labels)))
    except tf.errors.OutOfRangeError:
      sess.run(init_op)
  print(time.time() - tt)
