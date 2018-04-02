#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 11:55:41 2018

@author: hans
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import re
import arg_parsing

wd = arg_parsing.WEIGHT_DECAY # Weight decay
NUM_CLASSES = arg_parsing.NUM_LABELS

def _activation_summary(x,name):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('tower_[0-9]*/', '', name)
  tf.summary.image(tensor_name, tf.expand_dims(x[:,:,:,0], dim=3))
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _variable_with_weight_decay(shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal
    distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

    Returns:
      Variable Tensor
    """

    initializer = tf.truncated_normal_initializer(stddev=stddev)
    var = tf.get_variable('weights', shape=shape,
                          initializer=initializer)

    if wd and (not tf.get_variable_scope().reuse):
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def _bias_variable(shape, constant=0.0):
    initializer = tf.constant_initializer(constant)
    return tf.get_variable(name='biases', shape=shape,
                           initializer=initializer)

def _convolution_layer(bottom, shape, name):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
#        print('Layer name: %s' % name)
#        print('Layer shape: %s' % str(shape))
        
        # get number of input channels
        in_features = bottom.get_shape()[3].value
        out_features = shape[3]
#        print('In features: %s' % in_features)
#        print('Out features: %s' % out_features)
#        print('---------------------------------')

        # initialization
        stddev = (2 / (in_features + out_features))**0.5
        
        filt = _variable_with_weight_decay(shape, stddev, wd) 
        conv = tf.nn.conv2d(bottom, filt, strides=[1, 1, 1, 1], padding='SAME')
        
        conv_biases = _bias_variable([filt.get_shape()[3]], constant=0.0)
        bias = tf.nn.bias_add(conv, conv_biases)
        
        if name == 'conv10':
            out = bias
        else:
            out = tf.nn.elu(bias)
        
    # Add summary to Tensorboard
    _activation_summary(out,name)
    return out

def _max_pool(bottom, name):
    pool = tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name=name)
#    print('Layer name: %s' % name)
#    print('Layer shape:%s' % str(pool.get_shape()))
#    print('---------------------------------')
    _activation_summary(pool,name)
    return pool

def inference(images):
    """Build the model up to where it may be used for inference.
    Parameters
    ----------
    images : Images placeholder, from inputs().
    train : whether the network is used for train of inference
    Returns
    -------
    softmax_linear : Output tensor with the computed logits.
    """
    conv1 = _convolution_layer(images, [3,3,3,64], "conv1")
    pool1 = _max_pool(conv1, 'pool1')
    fire2_squeeze1x1 = _convolution_layer(pool1, [1,1,64,16], "fire2_squeeze1x1")
    fire2_expand1x1 = _convolution_layer(fire2_squeeze1x1, [1,1,16,64], "fire2_expand1x1")
    fire2_expand3x3 = _convolution_layer(fire2_squeeze1x1, [3,3,16,64], "fire2_expand3x3")
    fire2_concat = tf.concat([fire2_expand1x1, fire2_expand3x3], 3)
    fire3_squeeze1x1 = _convolution_layer(fire2_concat, [1,1,128,16], "fire3_squeeze1x1")
    fire3_expand1x1 = _convolution_layer(fire3_squeeze1x1, [1,1,16,64], "fire3_expand1x1")
    fire3_expand3x3 = _convolution_layer(fire3_squeeze1x1, [3,3,16,64], "fire3_expand3x3")
    fire3_concat = tf.concat([fire3_expand1x1, fire3_expand3x3], 3)
    pool3 = _max_pool(fire3_concat, 'pool3')
    fire4_squeeze1x1 = _convolution_layer(pool3, [1,1,128,128], "fire4_squeeze1x1")
    fire4_expand1x1 = _convolution_layer(fire4_squeeze1x1, [1,1,128,128], "fire4_expand1x1")
    fire4_expand3x3 = _convolution_layer(fire4_squeeze1x1, [3,3,128,128], "fire4_expand3x3")
    fire4_concat = tf.concat([fire4_expand1x1, fire4_expand3x3], 3)
    fire5_squeeze1x1 = _convolution_layer(fire4_concat, [1,1,256,128], "fire5_squeeze1x1")
    fire5_expand1x1 = _convolution_layer(fire5_squeeze1x1, [1,1,128,128], "fire5_expand1x1")
    fire5_expand3x3 = _convolution_layer(fire5_squeeze1x1, [3,3,128,128], "fire5_expand3x3")
    fire5_concat = tf.concat([fire5_expand1x1, fire5_expand3x3], 3)
    pool5 = _max_pool(fire5_concat, 'pool5')
    fire6_squeeze1x1 = _convolution_layer(pool5, [1,1,256,48], "fire6_squeeze1x1")
    fire6_expand1x1 = _convolution_layer(fire6_squeeze1x1, [1,1,48,192], "fire6_expand1x1")
    fire6_expand3x3 = _convolution_layer(fire6_squeeze1x1, [3,3,48,192], "fire6_expand3x3")
    fire6_concat = tf.concat([fire6_expand1x1, fire6_expand3x3], 3)
    fire7_squeeze1x1 = _convolution_layer(fire6_concat, [1,1,384,48], "fire7_squeeze1x1")
    fire7_expand1x1 = _convolution_layer(fire7_squeeze1x1, [1,1,48,192], "fire7_expand1x1")
    fire7_expand3x3 = _convolution_layer(fire7_squeeze1x1, [3,3,48,192], "fire7_expand3x3")
    fire7_concat = tf.concat([fire7_expand1x1, fire7_expand3x3], 3)
    fire8_squeeze1x1 = _convolution_layer(fire7_concat, [1,1,384,64], "fire8_squeeze1x1")
    fire8_expand1x1 = _convolution_layer(fire8_squeeze1x1, [1,1,64,256], "fire8_expand1x1")
    fire8_expand3x3 = _convolution_layer(fire8_squeeze1x1, [3,3,64,256], "fire8_expand3x3")
    fire8_concat = tf.concat([fire8_expand1x1, fire8_expand3x3], 3)
    fire9_squeeze1x1 = _convolution_layer(fire8_concat, [1,1,512,64], "fire9_squeeze1x1")
    fire9_expand1x1 = _convolution_layer(fire9_squeeze1x1, [1,1,64,256], "fire9_expand1x1")
    fire9_expand3x3 = _convolution_layer(fire9_squeeze1x1, [3,3,64,256], "fire9_expand3x3")
    fire9_concat = tf.concat([fire9_expand1x1, fire9_expand3x3], 3)
    drop9 = tf.nn.dropout(fire9_concat, keep_prob=0.5, name="drop9")
    conv10 = _convolution_layer(drop9, [1,1,512,NUM_CLASSES], "conv10")
    logits = tf.reduce_mean(conv10, [1,2], name='logits')
    return logits
