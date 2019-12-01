import sys
import os

train_imgs = set()
with open('train_info.csv') as fp:
  line = fp.readline()
  for line in fp:
    filename,label,loc_x_min,loc_x_max,loc_y_min,loc_y_max = line.strip().split(',')
    # if filename in train_imgs:
    #   print('boom! dup in train! name = {}'.format(filename))
    #   cnt += 1
    train_imgs.add(filename)


with open('test_info.csv') as fp:
  line = fp.readline()
  cnt = 0
  for line in fp:
    filename,label,loc_x_min,loc_x_max,loc_y_min,loc_y_max = line.strip().split(',')
    if filename in train_imgs:
      # print('{}'.format(filename))
      cnt += 1
  print('cnt = {}'.format(cnt))

# dir='./0'

# for root, dirs, files in os.walk(dir):
#   for filename in files:
#     if filename in train_imgs:
#       print('removing {}'.format(os.path.join(root, filename)))
#       os.remove(os.path.join(root, filename))


# dir='./1'

# for root, dirs, files in os.walk(dir):
#   for filename in files:
#     if filename in train_imgs:
#       print('removing {}'.format(os.path.join(root, filename)))
#       os.remove(os.path.join(root, filename))

test_imgs = set()
with open('test_info.csv') as fp:
  with open('test_info_final.csv', "w") as fpf:
    fpf.write("filename,label,loc_x_min,loc_x_max,loc_y_min,loc_y_max\n")
    line = fp.readline()
    for line in fp:
      filename,label,loc_x_min,loc_x_max,loc_y_min,loc_y_max = line.strip().split(',')
      if (not filename in train_imgs) and (not filename in test_imgs):
        fpf.write(filename + ',' + label + ',' + loc_x_min + ',' + loc_x_max + ',' + loc_y_min + ',' + loc_y_max + '\n')
      test_imgs.add(filename)

train_imgs = set()
with open('train_info.csv') as fp:
  with open('train_info_final.csv', "w") as fpf:
    fpf.write("filename,label,loc_x_min,loc_x_max,loc_y_min,loc_y_max\n")
    line = fp.readline()
    for line in fp:
      filename,label,loc_x_min,loc_x_max,loc_y_min,loc_y_max = line.strip().split(',')
      if not filename in train_imgs:
        fpf.write(filename + ',' + label + ',' + loc_x_min + ',' + loc_x_max + ',' + loc_y_min + ',' + loc_y_max + '\n')
      train_imgs.add(filename)


