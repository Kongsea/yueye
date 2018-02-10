#!/usr/bin/python
# -*- coding: utf-8 -*-
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

image_name = 'tiger'
image = Image.open('{}.jpg'.format(image_name))

color = 'red'
(left, top, right, bottom) = (270, 120, 420, 370)
class_name = '老虎'

try:
  font = ImageFont.truetype('wqy-microhei.ttc', 14)
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

image.show()
