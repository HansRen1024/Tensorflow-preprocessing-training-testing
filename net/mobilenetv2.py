#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 17:24:45 2018

@author: hans
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf
import tensorflow.contrib as tc
import arg_parsing
FLAGS = arg_parsing.parser.parse_args()
if FLAGS.mode == 'training_dis' or FLAGS.mode == 'training':
    is_training = True
else:
    is_training = False
normalizer = tc.layers.batch_norm
bn_params = {'is_training': is_training}

def _activation_summary(x,name):
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('tower_[0-9]*/', '', name)
    tf.summary.image(tensor_name, tf.expand_dims(x[:,:,:,0], dim=3))
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _inverted_bottleneck(input, up_sample_rate, channels, subsample,name):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE) :
        stride = 2 if subsample else 1
        output = tc.layers.conv2d(input, up_sample_rate*input.get_shape().as_list()[-1], 1,
                                  activation_fn=tf.nn.relu6,
                                  normalizer_fn=normalizer, normalizer_params=bn_params)
        output = tc.layers.separable_conv2d(output, None, 3, 1, stride=stride,
                                            activation_fn=tf.nn.relu6,
                                            normalizer_fn=normalizer, normalizer_params=bn_params)
        output = tc.layers.conv2d(output, channels, 1, activation_fn=None,
                                  normalizer_fn=normalizer, normalizer_params=bn_params)
        if input.get_shape().as_list()[-1] == channels:
            output = tf.add(input, output)
    _activation_summary(output,name)
    return output

def inference(images):
    with tf.variable_scope('conv0',reuse=tf.AUTO_REUSE) :
        output = tc.layers.conv2d(images, 32, 3, 2,normalizer_fn=normalizer, normalizer_params=bn_params)
    _activation_summary(output,'conv0')
    output = _inverted_bottleneck(output, 1, 16, 0, 'dw_conv1')
    output = _inverted_bottleneck(output, 6, 24, 1, 'dw_conv2')
    output = _inverted_bottleneck(output, 6, 24, 0, 'dw_conv3')
    output = _inverted_bottleneck(output, 6, 32, 1, 'dw_conv4')
    output = _inverted_bottleneck(output, 6, 32, 0, 'dw_conv5')
    output = _inverted_bottleneck(output, 6, 32, 0, 'dw_conv6')
    output = _inverted_bottleneck(output, 6, 64, 1, 'dw_conv7')
    output = _inverted_bottleneck(output, 6, 64, 0, 'dw_conv8')
    output = _inverted_bottleneck(output, 6, 64, 0, 'dw_conv9')
    output = _inverted_bottleneck(output, 6, 64, 0, 'dw_conv10')
    output = _inverted_bottleneck(output, 6, 96, 0, 'dw_conv11')
    output = _inverted_bottleneck(output, 6, 96, 0, 'dw_conv12')
    output = _inverted_bottleneck(output, 6, 96, 0, 'dw_conv13')
    output = _inverted_bottleneck(output, 6, 160, 1, 'dw_conv14')
    output = _inverted_bottleneck(output, 6, 160, 0, 'dw_conv15')
    output = _inverted_bottleneck(output, 6, 160, 0, 'dw_conv16')
    output = _inverted_bottleneck(output, 6, 320, 0, 'dw_conv17')
    with tf.variable_scope('conv18',reuse=tf.AUTO_REUSE) :
        output = tc.layers.conv2d(output, 1280, 1, normalizer_fn=normalizer, normalizer_params=bn_params)
    _activation_summary(output,'conv18')
    output = tc.layers.avg_pool2d(output, 7)
    with tf.variable_scope('conv19',reuse=tf.AUTO_REUSE) :
        output = tc.layers.conv2d(output, arg_parsing.NUM_LABELS, 1, activation_fn=None)
    _activation_summary(output,'conv19')
    logits = tf.reduce_mean(output, [1,2], name='logits')
    return logits
