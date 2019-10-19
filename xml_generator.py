import dicttoxml

import struct
import imghdr
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import subprocess
import math
import glob
import uuid
import scipy.io as sio

def get_image_size(fname):
  '''Determine the image type of fhandle and return its size.
  from draco'''
  with open(fname, 'rb') as fhandle:
    head = fhandle.read(24)
    if len(head) != 24:
      return
    if imghdr.what(fname) == 'png':
      check = struct.unpack('>i', head[4:8])[0]
      if check != 0x0d0a1a0a:
          return
      width, height = struct.unpack('>ii', head[16:24])
    elif imghdr.what(fname) == 'gif':
      width, height = struct.unpack('<HH', head[6:10])
    elif imghdr.what(fname) == 'jpeg':
      try:
        fhandle.seek(0) # Read 0xff next
        size = 2
        ftype = 0
        while not 0xc0 <= ftype <= 0xcf:
          fhandle.seek(size, 1)
          byte = fhandle.read(1)
          while ord(byte) == 0xff:
              byte = fhandle.read(1)
          ftype = ord(byte)
          size = struct.unpack('>H', fhandle.read(2))[0] - 2
        # We are at a SOFn block
        fhandle.seek(1, 1)  # Skip `precision' byte.
        height, width = struct.unpack('>HH', fhandle.read(4))
      except Exception: #IGNORE:W0703
        return
    else:
      return
    return width, height

def load_pascal_annotation(filename):
    """
    Load image and bounding boxes info from XML file in the PASCAL VOC
    format.
    """
    tree = ET.parse(filename)
    objs = tree.findall('object')[0]

    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    num_classes = 2
    classes = ('__background__',  # always index 0
                         'mine')
    class_to_ind = dict(zip(classes, range(num_classes)))

    overlaps = np.zeros((num_objs, num_classes), dtype=np.float32)
    # "Seg" area for pascal is just the box area
    seg_areas = np.zeros((num_objs), dtype=np.float32)
    ishards = np.zeros((num_objs), dtype=np.int32)

    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')

        # Make pixel indexes 0-based
        x1 = float(bbox.find('xmin').text) - 1
        y1 = float(bbox.find('ymin').text) - 1
        x2 = float(bbox.find('xmax').text) - 1
        y2 = float(bbox.find('ymax').text) - 1

        diffc = obj.find('difficult')
        difficult = 0 if diffc == None else int(diffc.text)
        ishards[ix] = difficult

        cls = class_to_ind[obj.find('name').text.lower().strip()]
        boxes[ix, :] = [x1, y1, x2, y2]
        gt_classes[ix] = cls
        overlaps[ix, cls] = 1.0
        seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

    overlaps = scipy.sparse.csr_matrix(overlaps)

    return {'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_ishard': ishards,
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': seg_areas}


with open('train_info_final.csv') as fp:
  with open('train.txt', 'w') as test_txt:
    
    line = fp.readline()
    cnt = 0
    for line in fp:
      filename,label,loc_x_min,loc_x_max,loc_y_min,loc_y_max = line.strip().split(',')
      path = str(label) + '/' + filename
      w, h = get_image_size(path)
      print('w = {}, h = {}, path = {}'.format(w, h, path))
      obj = {
        "folder": "VOC2007",
        "filename": "000001.jpg", # change here
        "source": {
            "database": "The VOC2007 Database",
            "annotation": "PASCAL VOC2007",
            "image": "flickr",
            "flickrid": "341012865"
        },
        "owner": {
            "flickrid": "none",
            "name": "Wenping Wang"
        },
        "size": {
            "width": "353", # change here
            "height": "500", # change here
            "depth": "3"
        },
        "segmented": "0",
        "object": [
          # {
          #   "name": "person", # change here
          #   "pose": "Left",
          #   "truncated": "1",
          #   "difficult": "0",
          #   "bndbox": {
          #       "xmin": "8", # change here
          #       "ymin": "12", # change here
          #       "xmax": "352", # change here
          #       "ymax": "498" # change here
          #   }
          # }
          ]
        
      }
      obj["filename"] = filename
      obj["size"]["width"] = str(w)
      obj["size"]["height"] = str(h)
      mine_box_num = 1

      loc_x_mins = loc_x_min.split('\t')
      loc_x_maxs = loc_x_max.split('\t')
      loc_y_mins = loc_y_min.split('\t')
      loc_y_maxs = loc_y_max.split('\t')
      mine_box_num = len(loc_y_maxs)

      for i in range(mine_box_num):
        obj_dict = {}

        obj_dict["name"] = "mine" if int(label) else '__background__'

        obj_dict["bndbox"] = {}
        obj_dict["bndbox"]["xmin"] = str(int(float(loc_x_mins[i]) * w))
        obj_dict["bndbox"]["xmax"] = str(int(float(loc_x_maxs[i]) * w))
        obj_dict["bndbox"]["ymin"] = str(int(float(loc_y_mins[i]) * h))
        obj_dict["bndbox"]["ymax"] = str(int(float(loc_y_maxs[i]) * h))

        obj["object"].append(obj_dict)

      # print(obj)

      xml = dicttoxml.dicttoxml(obj)
      # print(xml)

      with open(filename.split('.jpg')[0] + '.xml', 'wb') as xml_file:
        xml_file.write(xml)

      parsed = load_pascal_annotation(filename.split('.jpg')[0] + '.xml')
      print(parsed)

      # print(parsed['boxes'])
      # print(parsed['gt_classes'])
      # print(parsed['gt_ishard'])
      # print(parsed['gt_overlaps'])
      # print(parsed['seg_areas'])
      cnt += 1

      # exit()
      test_txt.write(filename.split('.jpg')[0] + '\n')
    print('cnt = {}'.format(cnt))

