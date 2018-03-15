#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 10:25:31 2018

@author: hans
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

import arg_parsing
import train
import test

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def main(argv=None):

    FLAGS = arg_parsing.parser.parse_args()

    if (FLAGS.mode == 'training'):
#        if tf.gfile.Exists(FLAGS.model_dir):
#            tf.gfile.DeleteRecursively(FLAGS.model_dir)
#        tf.gfile.MakeDirs(FLAGS.model_dir)
        train.train_dis_()

    elif (FLAGS.mode == 'testing'):
        test.test(FLAGS.mode)
    else:
        raise ValueError("set --mode as 'training' or 'testing'")

if __name__ == '__main__':
    tf.app.run()
