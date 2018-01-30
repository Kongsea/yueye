#!/usr/bin/python
# -*- coding: utf-8 -*-
import time

import cv2
import tensorflow as tf


label_map = {'猫': 0, '狗': 1}


def gen():
  with open('train.csv') as f:
    lines = [line.strip().split(',') for line in f.readlines()]

  index = 0
  while True:
    image = cv2.imread(lines[index][0])
    image = cv2.resize(image, (224, 224))
    label = label_map[lines[index][1]]
    yield (image, label)
    index += 1
    if index == len(lines):
      index = 0


def create_dataset():

  data = tf.data.Dataset.from_generator(gen, (tf.float32, tf.int32),
                                        (tf.TensorShape([224, 224, 3]), tf.TensorShape([])))
  data = data.batch(2)
  data = data.make_one_shot_iterator()
  tt = time.time()
  with tf.Session() as sess:
    for i in range(100):
      _, labels = data.get_next()
      labels = sess.run(labels)
      print('{} -> {}'.format(i, labels))
    print(time.time() - tt)


def main():
  create_dataset()


if __name__ == '__main__':
  main()
