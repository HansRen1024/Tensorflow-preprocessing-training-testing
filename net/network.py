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
import arg_parsing

def _conv_module(inputs, num_outputs, kernel_size, stride, scope, activation_fn=tf.nn.relu, pad='SAME'):
    with tf.variable_scope(scope, 'Conv', [inputs]) as sc:
        kSize = kernel_size+[inputs.get_shape().as_list()[3], num_outputs]
        kernel = tf.Variable(tf.random_normal(kSize))
        conv = tf.nn.conv2d(inputs, kernel, [1, stride, stride, 1], padding=pad)
        biases = tf.Variable(tf.random_normal([num_outputs]))
        conv1 = tf.nn.bias_add(conv, biases)
        if activation_fn is not None:
            conv1 = activation_fn(conv1, name=sc.name)
    return conv1

def _fire_module(inputs, squeeze_depth, expand_depth, scope):
    with tf.variable_scope(scope, 'Fire', [inputs]) as sc:
        net = _conv_module(inputs, squeeze_depth, [1, 1], 1, scope='squeeze')
        with tf.variable_scope('expand'):
            e1x1 = _conv_module(net, expand_depth, [1, 1], 1, scope='1x1')
            e3x3 = _conv_module(net, expand_depth, [3, 3], 1, scope='3x3')
            net = tf.concat([e1x1, e3x3], 3)
    net = tf.clip_by_norm(net, 100)
    return net

def inference(images,train=False):
    net = _conv_module(images, 64, [3, 3], 2, 'conv1')
    net = tf.nn.max_pool(net, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')
    net = _fire_module(net, 16, 64, 'fire2')
    net = _fire_module(net, 16, 64, 'fire3')
    net = tf.nn.max_pool(net, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')
    net = _fire_module(net, 32, 128, 'fire4')
    net = _fire_module(net, 32, 128, 'fire5')
    net = tf.nn.max_pool(net, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')
    net = _fire_module(net, 48, 192, 'fire6')
    net = _fire_module(net, 48, 192, 'fire7')
    net = _fire_module(net, 64, 256, 'fire8')
    net = _fire_module(net, 64, 256, 'fire9')
    if train:
        net = tf.nn.dropout(net, 0.5, noise_shape=None, seed=None,name=None) 
    net = _conv_module(net, arg_parsing.NUM_LABELS, [1, 1], 1, 'conv10', None)
    logits = tf.reduce_mean(net, [1,2], name='logits')
#    logits = tf.squeeze(net, [1, 2], name='logits')
    return logits
