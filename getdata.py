# feature = {'image/width': tf.FixedLenFeature([], dtype=tf.int64),
#             'image/object/class/label': tf.VarLenFeature(dtype=tf.int64),
#             'image/height': tf.FixedLenFeature([], dtype=tf.int64),
#             'image/object/class/text': tf.VarLenFeature(dtype=tf.string),
#             'image/source_id': tf.VarLenFeature(dtype=tf.string),
#             'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
#             'image/encoded': tf.VarLenFeature(dtype=tf.string)}


import tensorflow as tf
from PIL import Image
import numpy as np
import os
from io import BytesIO

def train_input_fn():
  filenames = ["./training.record"]
  dataset = tf.data.TFRecordDataset(filenames)

  def parser(record):
    keys_to_features = {'image/width': tf.FixedLenFeature((), dtype=tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
            'image/object/class/label': tf.FixedLenFeature((), dtype=tf.int64),
            'image/height': tf.FixedLenFeature([], dtype=tf.int64),
            'image/object/class/text': tf.VarLenFeature(dtype=tf.string),
            'image/source_id': tf.VarLenFeature(dtype=tf.string),
            'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
            'image/encoded': tf.FixedLenFeature((), dtype=tf.string, default_value="")}
    
    parsed = tf.parse_single_example(record, keys_to_features)

    # image = tf.decode_jpg(parsed["image/encoded"])
    image = parsed["image/encoded"]

    label = parsed["image/object/class/label"]

    width = parsed["image/width"]
    height = parsed["image/height"]
    source_id = parsed['image/source_id']
    text = parsed['image/object/class/text']
    ymin = parsed['image/object/bbox/ymin']
    # label = tf.cast(parsed["image/object/class/label"], tf.int32)
    

    return image, label, source_id, width, height, text, ymin

  dataset = dataset.map(parser)

  # dataset = dataset.shuffle(buffer_size=10000)
  # dataset = dataset.batch(32)
  # dataset = dataset.repeat(10)
  iterator = dataset.make_one_shot_iterator()

  # image, label, source_id, width, height, text, ymin = iterator.get_next()
  # return image, label, source_id, width, height, text, ymin
  next_element = iterator.get_next()
  return next_element

sess = tf.InteractiveSession()


# image, label, source_id, width, height, text, ymin = train_input_fn()
next_element = train_input_fn()

image, label, source_id, width, height, text, ymin = sess.run(next_element)

# image = sess.run(image)
# label = sess.run(label)
# source_id = sess.run(source_id)
# width = sess.run(width)
# height = sess.run(height)
# text = sess.run(text)
# ymin = sess.run(ymin)



print(type(image))
# print(image)
img = Image.open(BytesIO(image))
w, h = img.size
print('w = ' + str(w) + ', h = ' + str(h))
print('size = ' + str(np.array(img).shape))
img.save(source_id.values[0].decode("utf-8"))


print('label:')
print(type(label))
print(label)

print('source_id:')
print(type(source_id))
print(source_id.values[0].decode("utf-8"))

print('width:')
print(type(width))
print(width.size)
print(width)

print('height:')
print(type(height))
print(height.size)
print(height)

print('text:')
print(type(text))
print(text)

print('ymin:')
print(type(ymin))
print(ymin)


for i in range(100):
  image, label, source_id, width, height, text, ymin = sess.run(next_element)
  print('label ' + str(i + 1))
  print(type(label))
  print(label)

exit()
