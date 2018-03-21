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

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 27013
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 4996
NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 4996
NUM_LABELS = 17
IMAGE_SHAPE = 227
mean = [82.4088, 114.4588, 139.1493] # BGR

INITIAL_LEARNING_RATE = 0.01
MOVING_AVERAGE_DECAY = 0.9999
NUM_EPOCHS_PER_DECAY = 50
LEARNING_RATE_DECAY_FACTOR = 0.75
STEPS_TO_VAL = 1000
WEIGHT_DECAY = 2e-4
b = tf.Variable(mean[0], dtype=tf.float32, name='b', trainable=False)
g = tf.Variable(mean[1], dtype=tf.float32, name='g', trainable=False)
r = tf.Variable(mean[2], dtype=tf.float32, name='r', trainable=False)
MEAN = [b, g, r]

DEBUG = False
DATASET_DIR = 'data/' # Path to data directory.
MODEL_DIR = 'models/' # Directory where to write event logs and checkpoint.
BATCH_SIZE = 16
LOG_FREQUENCY = 100 # How often to log results to the console.
MAX_STEPS = 10000 # Number of batches to run.
LOG_DEVICE_PLACEMENT = False # Whether to log device placement.
USE_FP16 = False # Train the model using fp16.

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str,
                    help='Either `training` or `testing` or `training_dis` .')

# For distributed
PS_HOSTS = '10.100.1.100:2222' # Comma-separated list of hostname:port pairs
WORKER_HOSTS = '10.100.1.100:2224,10.100.1.101:2225' # Comma-separated list of hostname:port pairs

parser.add_argument("--job_name", type=str,
                    help="One of 'ps', 'worker'")
parser.add_argument("--task_index", type=int,
                    help="Index of task within the job")
