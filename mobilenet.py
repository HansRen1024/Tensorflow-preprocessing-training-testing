#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 11:33:52 2018

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

def _dw_conv(input,st,num_outputs,name):
    dconv = tc.layers.separable_conv2d(input,None,3,1,st,activation_fn=tf.nn.relu6,
                                       normalizer_fn=normalizer, normalizer_params=bn_params)
    pconv = tc.layers.conv2d(dconv, num_outputs, 1, normalizer_fn=normalizer, normalizer_params=bn_params)
    _activation_summary(pconv,name)
    return pconv

def inference(images):
    output = tc.layers.conv2d(images, num_outputs=32, kernel_size=3, stride=2,
                                             normalizer_fn=normalizer, normalizer_params=bn_params)
    _activation_summary(output,'conv0')
    output = _dw_conv(output,1,64,'pconv1')
    output = _dw_conv(output,2,128,'pconv2')
    output = _dw_conv(output,1,128,'pconv3')
    output = _dw_conv(output,2,256,'pconv4')
    output = _dw_conv(output,1,256,'pconv5')
    output = _dw_conv(output,2,512,'pconv6')
    output = _dw_conv(output,1,512,'pconv71')
    output = _dw_conv(output,1,512,'pconv72')
    output = _dw_conv(output,1,512,'pconv73')
    output = _dw_conv(output,1,512,'pconv74')
    output = _dw_conv(output,1,512,'pconv75')
    output = _dw_conv(output,2,1024,'pconv8')
    output = _dw_conv(output,1,1024,'pconv9')
    output = tc.layers.avg_pool2d(output, kernel_size=7, stride=1)
    tf.summary.image("pool", tf.expand_dims(output[:,:,:,0], dim=3))
    output = tc.layers.conv2d(output, arg_parsing.NUM_LABELS, 1, activation_fn=None)
    _activation_summary(output,'conv10')
    logits = tf.reduce_mean(output, [1,2], name='logits')
    return logits
