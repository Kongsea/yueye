#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

DIGITS = 20


def binary_encode(num, digits=DIGITS):
  return [num >> i & 1 for i in range(digits)][::-1]


def label_encode(num):
  if num % 15 == 0:
    return [1, 0, 0, 0]
  elif num % 3 == 0:
    return [0, 1, 0, 0]
  elif num % 5 == 0:
    return [0, 0, 1, 0]
  else:
    return [0, 0, 0, 1]


def get_data(num, low=101, high=10000):
  binary_num_list = []
  label_list = []
  for i in range(num):
    n = np.random.randint(low, high, 1)[0]
    binary_num_list.append(np.array(binary_encode(n)))
    label_list.append(np.array(label_encode(n)))
  return np.array(binary_num_list), np.array(label_list)


def model(data):
  with tf.variable_scope('layer1') as scope:
    weight = tf.get_variable('weight', shape=(DIGITS, 256))
    bias = tf.get_variable('bias', shape=(256,))
    x = tf.nn.relu(tf.matmul(data, weight) + bias)

  with tf.variable_scope('layer2') as scope:
    weight = tf.get_variable('weight', shape=(256, 4))
    bias = tf.get_variable('bias', shape=(4,))
    x = tf.matmul(x, weight) + bias

  return x


def main():
  data = tf.placeholder(tf.float32, shape=(None, DIGITS))
  label = tf.placeholder(tf.float32, shape=(None, 4))

  x = model(data)
  preds = tf.argmax(tf.nn.softmax(x), 1)
  acc = tf.reduce_mean(tf.cast(tf.equal(preds, tf.argmax(label, 1)), tf.float32))
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=x))
  optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(3000):
      train_data, train_label = get_data(128)
      _, a = sess.run([optimizer, acc],
                      feed_dict={data: train_data, label: train_label})
      if step % 300 == 0:
        print('Step: {} -> Accuracy: {:.3f}'.format(step, a))

    test_data = np.array([binary_encode(i) for i in range(1, 101)])
    pred = sess.run(preds, feed_dict={data: test_data})
    results = []
    for i in range(1, 101):
      results.append('{}'.format(['fizzbuzz', 'fizz', 'buzz', i][pred[i - 1]]))
    print(', '.join(results))


if __name__ == '__main__':
  main()
