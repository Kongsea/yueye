#!/usr/bin/python
# -*- coding: utf-8 -*-
import json
import os
import sys

import cv2

if len(sys.argv) < 3:
  print "Usage: python convert_to_detectron_json.py root_path phase split"
  print "For example: python convert_to_detectron_json.py data train 100200"
  exit(1)

root_path = sys.argv[1]
phase = sys.argv[2]
split = int(sys.argv[3])

dataset = {
    'licenses': [],
    'info': {},
    'categories': [],
    'images': [],
    'annotations': []
}

with open(os.path.join(root_path, 'classes.txt')) as f:
  classes = f.read().strip().split()

for i, cls in enumerate(classes, 1):
  dataset['categories'].append({
      'id': i,
      'name': cls,
      'supercategory': 'beverage'
  })


def get_category_id(cls):
  for category in dataset['categories']:
    if category['name'] == cls:
      return category['id']


_indexes = sorted([f.split('.')[0] for f in os.listdir(os.path.join(root_path, 'annos'))])
if phase == 'train':
  indexes = [line for line in _indexes if int(line) > split]
else:
  indexes = [line for line in _indexes if int(line) <= split]

j = 1
for index in indexes:
  im = cv2.imread(os.path.join(root_path, 'images/') + index + '.jpg')
  height, width, _ = im.shape
  dataset['images'].append({
      'coco_url': '',
      'date_captured': '',
      'file_name': index + '.jpg',
      'flickr_url': '',
      'id': int(index),
      'license': 0,
      'width': width,
      'height': height
  })

  anno_file = os.path.join(root_path, 'annos/') + index + '.txt'
  with open(anno_file) as f:
    lines = [line for line in f.readlines() if line.strip()]

    for i, line in enumerate(lines):
      parts = line.strip().split()
      cls = parts[0]
      x1 = int(parts[1])
      y1 = int(parts[2])
      x2 = int(parts[3])
      y2 = int(parts[4])
      width = max(0, x2 - x1)
      height = max(0, y2 - y1)
      dataset['annotations'].append({
          'area': width * height,
          'bbox': [x1, y1, width, height],
          'category_id': get_category_id(cls),
          'id': j,
          'image_id': int(index),
          'iscrowd': 0,
          'segmentation': []
      })
      j += 1

folder = os.path.join(root_path, 'annotations')
if not os.path.exists(folder):
  os.makedirs(folder)
json_name = os.path.join(root_path, 'annotations/{}.json'.format(phase))
with open(json_name, 'w') as f:
  json.dump(dataset, f)
