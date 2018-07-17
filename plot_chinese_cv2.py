#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

image_name = 'tiger'
im = cv2.imread('{}.jpg'.format(image_name))
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
image = Image.fromarray(im)

color = 'red'
(left, top, right, bottom) = (270, 120, 420, 370)
class_name = '老虎'

try:
  font = ImageFont.truetype('wqy-microhei.ttc', 18)
except IOError:
  font = ImageFont.load_default()

draw = ImageDraw.Draw(image)
draw.line([(left, top), (left, bottom), (right, bottom),
           (right, top), (left, top)], width=1, fill=color)

draw.text(
    (left, top-14),
    class_name.decode('utf-8'),
    fill='black',
    font=font)

im = np.array(image)
im = im[:, :, ::-1].copy()

cv2.imwrite('test.jpg', im)
