import tensorflow as tf
from google.protobuf.json_format import MessageToJson
import json
from PIL import Image
import numpy as np
import os
from io import BytesIO

label_path = {
  1: './1/',
  2: './0/'
}

f = open("info.csv", "w")
f.write("filename,label,loc_x_min,loc_x_max,loc_y_min,loc_y_max\n")
for i, example in enumerate(tf.python_io.tf_record_iterator("./training.record")):
  result = tf.train.Example.FromString(example)

  # for key in result.features.feature:
  #   if key == 'image/object/class/label':
  #     print(result.features.feature[key].int64_list.value[0])

  image = result.features.feature['image/encoded'].bytes_list.value[0]
  filename = result.features.feature['image/filename'].bytes_list.value[0]
  label = result.features.feature['image/object/class/label'].int64_list.value[0]
  
  locs = result.features.feature['image/object/bbox/xmin'].float_list.value
  loc_x_min = '\t'.join(str(loc) for loc in locs)

  locs = result.features.feature['image/object/bbox/xmax'].float_list.value
  loc_x_max = '\t'.join(str(loc) for loc in locs)

  locs = result.features.feature['image/object/bbox/ymin'].float_list.value
  loc_y_min = '\t'.join(str(loc) for loc in locs)

  locs = result.features.feature['image/object/bbox/ymax'].float_list.value
  loc_y_max = '\t'.join(str(loc) for loc in locs)

  f.write(filename.decode("utf-8") + ',' + str(label) + ',' + loc_x_min + ',' + loc_x_max + ',' + loc_y_min + ',' + loc_y_max + '\n')


  img = Image.open(BytesIO(image))
  # w, h = img.size
  # print('w = ' + str(w) + ', h = ' + str(h))
  # print('size = ' + str(np.array(img).shape))
  img.save(label_path[label] + str(i) + filename.decode("utf-8"))



  # jsonMessage = MessageToJson(result)
  # python_obj = json.loads(jsonMessage)
  # print(python_obj['features']['feature']['image/object/class/label']['int64List'])

