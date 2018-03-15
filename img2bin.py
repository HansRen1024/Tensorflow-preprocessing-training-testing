import os
import tensorflow as tf 
from PIL import Image

cwd = os.getcwd()
classes=['0','1','2']
writer = tf.python_io.TFRecordWriter("train.tfrecords")
num=1;
for index, name in enumerate(classes):
    class_path = cwd + '/ke/' + name + "/"
    for img_name in os.listdir(class_path):
        img_path = class_path + img_name
        print('%d dealing with %s' %(num,img_path))
        img = Image.open(img_path)
        img = img.resize((227, 227))
        img_raw = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())
        num=num+1
writer.close()
