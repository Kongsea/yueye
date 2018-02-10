#!/usr/bin/python
# -*- coding: utf-8 -*-
import random

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.sans-serif'] = ['SimHei']


image_name = 'tiger'
im = cv2.imread('{}.jpg'.format(image_name))
im = im[:, :, (2, 1, 0)]
_, ax = plt.subplots()
ax.imshow(im, aspect='equal')
ax.set_title(('detection result for {}').format(image_name), fontsize=14)

bbox = [270, 120, 420, 370]
score = 0.99

color = (random.random(), random.random(), random.random())
ax.add_patch(
    plt.Rectangle((bbox[0], bbox[1]),
                  bbox[2] - bbox[0],
                  bbox[3] - bbox[1], fill=False,
                  edgecolor=color, linewidth=1)
)
class_name = '老虎'
ax.text(bbox[0], bbox[1] - 2,
        u'{:s} {:.3f}'.format(class_name.decode('utf-8'), score),
        # unicode(class_name.decode('utf-8')) + str(score),
        bbox=dict(facecolor='blue', alpha=0.5),
        fontsize=14, color='white')

plt.axis('off')
plt.tight_layout()
plt.draw()
plt.show()
