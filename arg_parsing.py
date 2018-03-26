#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 10:25:31 2018

@author: hans
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

import argparse

NET = 'squeezenet' # squeezenet or mobilenet or mobilenetv2 or resnet ...
RESNET_LAYER_NUM = 50 # 50 or 101 or 152 or 200 or ...
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 27013 # number of training data
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 4996 # number of validation data 
NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 4996 # number of testing data
NUM_LABELS = 17 # number of classes
ORIGIN_IMAGE_SHAPE = 227 # origin image shape that equal to size in img2bin_list.py
mean = [82.4088, 114.4588, 139.1493] # BGR mean values

INITIAL_LEARNING_RATE = 0.01
MOVING_AVERAGE_DECAY = 0.9999
NUM_EPOCHS_PER_DECAY = 10
LEARNING_RATE_DECAY_FACTOR = 0.5
STEPS_TO_VAL = 1000
WEIGHT_DECAY = 2e-4 # used for squeezenet and resnet

DEBUG = False
DATASET_DIR = 'data/' # Path to data directory.
MODEL_DIR = 'models/' # Directory where to write event logs and checkpoint.
STEPS_TO_SAVE_MODEL=1000
BATCH_SIZE = 4
LOG_FREQUENCY = 100 # How often to log results to the console.
MAX_STEPS = 10000 # Number of batches to run. If distributiong, all GPU batches.
LOG_DEVICE_PLACEMENT = False # Whether to log device placement.
USE_FP16 = False # Train the model using fp16.

if NET == 'squeezenet':
    IMAGE_RESIZE_SHAPE = 227 # image shape which suits for network
elif NET == 'mobilenet' or NET == 'mobilenetv2' or NET == 'resnet':
    IMAGE_RESIZE_SHAPE = 224

b = tf.Variable(mean[0], dtype=tf.float32, name='b', trainable=False)
g = tf.Variable(mean[1], dtype=tf.float32, name='g', trainable=False)
r = tf.Variable(mean[2], dtype=tf.float32, name='r', trainable=False)
MEAN = [b, g, r]
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str,
                    help='Either `training` or `testing` or `training_dis` .')
parser.add_argument('--lr', type=int, default=INITIAL_LEARNING_RATE)
parser.add_argument('--debug', type=bool, default=DEBUG)

# For distributed
PS_HOSTS = '10.100.3.101:2222' # Comma-separated list of hostname:port pairs
WORKER_HOSTS = '10.100.3.101:2224,10.100.3.100:2225' # Comma-separated list of hostname:port pairs

parser.add_argument("--job_name", type=str,
                    help="One of 'ps', 'worker'")
parser.add_argument("--task_index", type=int,
                    help="Index of task within the job")
