#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 10:25:31 2018

@author: hans
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 27013
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 4996
NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 10000
NUM_LABELS = 17

MOVING_AVERAGE_DECAY = 0.9999
NUM_EPOCHS_PER_DECAY = 50
LEARNING_RATE_DECAY_FACTOR = 0.75
INITIAL_LEARNING_RATE = 0.01
STEPS_TO_VAL = 1000
WEIGHT_DECAY = 2e-4


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str,
                    help='Either `training` or `testing` or `training_dis` .')
parser.add_argument('--data_dir', type=str, default='data/',
                    help='Path to data directory.')
parser.add_argument('--model_dir', type=str, default='models/',
                    help='Directory where to write event logs and checkpoint.')
parser.add_argument('--batch_size', type=int, default=1,
                    help='Number of images to process in a batch.')
parser.add_argument('--use_fp16', type=bool, default=False,
                    help='Train the model using fp16.')
parser.add_argument('--max_steps', type=int, default=10000,
                    help='Number of batches to run.')
parser.add_argument('--log_frequency', type=int, default=100,
                    help='How often to log results to the console.')
parser.add_argument('--log_device_placement', type=bool, default=False,
                    help='Whether to log device placement.')
parser.add_argument('--debug', type=bool, default=False,
                    help='Whether to debug.')

# For distributed
parser.add_argument("--ps_hosts", type=str,
                    help="Comma-separated list of hostname:port pairs")
parser.add_argument("--worker_hosts", type=str,
                    help="Comma-separated list of hostname:port pairs")
parser.add_argument("--job_name", type=str,
                    help="One of 'ps', 'worker'")
parser.add_argument("--task_index", type=int, default=0,
                    help="Index of task within the job")
