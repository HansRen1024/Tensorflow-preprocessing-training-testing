#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 10:25:31 2018

@author: hans
"""


import os
import tensorflow as tf 
from PIL import Image

file_path = 'train.txt'

root_path = '/home/hans/caffe/Home/image/'
writer_name = file_path.split('.',1)[0]
writer = tf.python_io.TFRecordWriter(writer_name+".tfrecords")
num=1;

if not os.path.isfile(file_path):
    raise TypeError(file_path + " does not exist")
with open(file_path) as f:
    line = f.readline()
    while line:
        name = line.split(' ', 1)[0]
        index = int(line.split(' ', 1)[1])
        print('%d dealing with %s' %(num,name))
        img_path = root_path + name
        if not os.path.isfile(img_path):
            line = f.readline()
            continue
        img = Image.open(img_path)
        img = img.resize((227, 227))
        img = img.convert('RGB')
        img_raw = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))
        writer.write(example.SerializeToString())
        num=num+1
        line = f.readline()
writer.close()
