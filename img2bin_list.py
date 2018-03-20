#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 10:25:31 2018

@author: hans
"""


import os
import tensorflow as tf 
import cv2

mode='test' # train or val or test
root_path='/home/hans/caffe/Home/fruit-2.0/image/' 
size=227

file_path = mode+'.txt'
writer = tf.python_io.TFRecordWriter('data/' + mode + ".tfrecords")
num=1;
b_mean_total = 0
g_mean_total = 0
r_mean_total = 0

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
        
#        from PIL import Image
#        img = Image.open(img_path)
#        img = img.resize((size, size))
#        img = img.convert('RGB')
#        img_raw = img.tobytes()
        
        image = cv2.imread(img_path)
        image = cv2.resize(image,(size,size))
        b,g,r = cv2.split(image)
        
        b_mean_total += b.mean()
        g_mean_total += g.mean()
        r_mean_total += r.mean()
        
        rgb_image = cv2.merge([r,g,b])
        img_raw = rgb_image.tostring()
        
        example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))
        writer.write(example.SerializeToString())
        num=num+1
        line = f.readline()
writer.close()
print('>>>>BGR mean values: [%.4f, %.4f, %.4f]' %(b_mean_total/float(num), g_mean_total/float(num), r_mean_total/float(num)))
