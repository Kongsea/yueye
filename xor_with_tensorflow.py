#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

data = tf.placeholder(tf.float32, shape=(4, 2))
label = tf.placeholder(tf.float32, shape=(4, 1))

with tf.variable_scope('layer1') as scope:
  weight = tf.get_variable(name='weight', shape=(4, 2))
  bias = tf.get_variable(name='bias', shape=(2,))
  x = tf.nn.sigmoid(tf.matmul(data, weight) + bias)
with tf.variable_scope('layer2') as scope:
  weight = tf.get_variable(name='weight', shape=(2, 1))
  bias = tf.get_variable(name='bias', shape=(1,))
  x = tf.matmul(x, weight) + bias

preds = tf.nn.sigmoid(x)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=x))
learning_rate = tf.placeholder(tf.float32)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

train_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
train_label = np.array([[0], [1], [1], [0]])

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for step in range(10000):
    if step < 3000:
      lr = 1
    elif step < 6000:
      lr = 0.1
    else:
      lr = 0.01
    _, l, pred = sess.run([optimizer, loss, preds], feed_dict={
                          data: train_data, label: train_label, learning_rate: lr})
    if step % 500:
      print('Step: {} -> Loss: {} -> Predictions: {}'.format(step, l, pred))
