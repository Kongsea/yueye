#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from skimage import measure


def write(image, path='test.png'):
  cv2.imwrite(path, image)


def draw_mask_edge_on_image_cv2(image, mask, color=(0, 0, 255)):
  coef = 255 if np.max(image) < 3 else 1
  image = (image * coef).astype(np.float32)
  contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
  cv2.drawContours(image, contours, -1, color, 1)
  write(image)


def draw_mask_on_image(image, mask, color=(0, 0, 255)):
  coef = 255 if np.max(image) < 3 else 1
  image = (image * coef).astype(np.float32)
  contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
  cv2.drawContours(image, contours, -1, color, -1)
  write(image)


def draw_mask_edge_on_image_skimage(image, mask, color=(0, 0, 255)):
  coef = 255 if np.max(image) < 3 else 1
  image = (image * coef).astype(np.float32)
  contours = measure.find_contours(mask, 0.5)
  image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
  for c in contours:
    c = np.around(c).astype(np.int)
    image[c[:, 0], c[:, 1]] = np.array(color)
  write(image)


def draw_mask_edge_on_skimage_using_cv2(image, mask, color=(0, 0, 255)):
  coef = 255 if np.max(image) < 3 else 1
  image = (image * coef).astype(np.float32)
  contours = measure.find_contours(mask, 0.5)
  image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
  _contours = []
  for c in contours:
    c[:, [0, 1]] = c[:, [1, 0]]
    _contours.append(np.around(np.expand_dims(c, 1)).astype(np.int))
  cv2.drawContours(image, _contours, -1, color, 1)
  write(image)


def draw_mask_on_image_cv2(image, mask):
  coef = 255 if np.max(image) < 3 else 1
  image = (image * coef).astype(np.float32)
  image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
  image[..., 2] = np.where(mask == 255, 255, image[..., 2])
  write(image)


def main():
  image = cv2.imread('image.png', 0)
  mask = cv2.imread('mask.png', 0)

  write(image)
  write(mask)

  mask = np.where(mask > 150, 255, np.where(mask < 100, 0, mask))

  draw_mask_on_image_cv2(image, mask)

  draw_mask_edge_on_image_cv2(image, mask)

  draw_mask_edge_on_image_skimage(image, mask)

  draw_mask_edge_on_skimage_using_cv2(image, mask)

  draw_mask_on_image(image, mask)


if __name__ == '__main__':
  main()
