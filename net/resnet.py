#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 10:34:33 2018

@author: hans
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf
import sys
sys.path.append("..")
import arg_parsing
slim = tf.contrib.slim
num_classes = arg_parsing.NUM_LABELS
FLAGS = arg_parsing.parser.parse_args()
if FLAGS.mode == 'training_dis' or FLAGS.mode == 'training':
    is_training = True
else:
    is_training = False

global_pool=True

class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    """A named tuple describing a ResNet block.
    Its parts are:
    scope: The scope of the `Block`.
    unit_fn: The ResNet unit function which takes as input a `Tensor` and
    returns another `Tensor` with the output of the ResNet unit.
    args: A list of length equal to the number of units in the `Block`. The list
    contains one (depth, depth_bottleneck, stride) tuple for each unit in the
    block to serve as argument to unit_fn.
    """


def subsample(inputs, factor, scope=None):
    """Subsamples the input along the spatial dimensions.
    Args:
        inputs: A `Tensor` of size [batch, height_in, width_in, channels].
        factor: The subsampling factor.
        scope: Optional variable_scope.
    Returns:
        output: A `Tensor` of size [batch, height_out, width_out, channels] with the
        input, either intact (if factor == 1) or subsampled (if factor > 1).
    """
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)


def conv2d_same(inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
    """Strided 2-D convolution with 'SAME' padding.
    When stride > 1, then we do explicit zero-padding, followed by conv2d with
    'VALID' padding.
    Note that
        net = conv2d_same(inputs, num_outputs, 3, stride=stride)
    is equivalent to
        net = slim.conv2d(inputs, num_outputs, 3, stride=1, padding='SAME')
        net = subsample(net, factor=stride)
    whereas
        net = slim.conv2d(inputs, num_outputs, 3, stride=stride, padding='SAME')
    is different when the input's height or width is even, which is why we add the
    current function. For more details, see ResnetUtilsTest.testConv2DSameEven().
    Args:
        inputs: A 4-D tensor of size [batch, height_in, width_in, channels].
        num_outputs: An integer, the number of output filters.
        kernel_size: An int with the kernel_size of the filters.
        stride: An integer, the output stride.
        rate: An integer, rate for atrous convolution.
        scope: Scope.
    Returns:
        output: A 4-D tensor of size [batch, height_out, width_out, channels] with
        the convolution output.
    """
    if stride == 1:
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=1, rate=rate,
                       padding='SAME', scope=scope)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs,[[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                       rate=rate, padding='VALID', scope=scope)

@slim.add_arg_scope
def stack_blocks_dense(net, blocks, output_stride=None,
                       outputs_collections=None):
      """Stacks ResNet `Blocks` and controls output feature density.
      First, this function creates scopes for the ResNet in the form of
      'block_name/unit_1', 'block_name/unit_2', etc.
      Second, this function allows the user to explicitly control the ResNet
      output_stride, which is the ratio of the input to output spatial resolution.
      This is useful for dense prediction tasks such as semantic segmentation or
      object detection.
      Most ResNets consist of 4 ResNet blocks and subsample the activations by a
      factor of 2 when transitioning between consecutive ResNet blocks. This results
      to a nominal ResNet output_stride equal to 8. If we set the output_stride to
      half the nominal network stride (e.g., output_stride=4), then we compute
      responses twice.
      Control of the output feature density is implemented by atrous convolution.
      Args:
        net: A `Tensor` of size [batch, height, width, channels].
        blocks: A list of length equal to the number of ResNet `Blocks`. Each
          element is a ResNet `Block` object describing the units in the `Block`.
        output_stride: If `None`, then the output will be computed at the nominal
          network stride. If output_stride is not `None`, it specifies the requested
          ratio of input to output spatial resolution, which needs to be equal to
          the product of unit strides from the start up to some level of the ResNet.
          For example, if the ResNet employs units with strides 1, 2, 1, 3, 4, 1,
          then valid values for the output_stride are 1, 2, 6, 24 or None (which
          is equivalent to output_stride=24).
        outputs_collections: Collection to add the ResNet block outputs.
      Returns:
        net: Output tensor with stride equal to the specified output_stride.
      Raises:
        ValueError: If the target output_stride is not valid.
      """
      # The current_stride variable keeps track of the effective stride of the
      # activations. This allows us to invoke atrous convolution whenever applying
      # the next residual unit would result in the activations having stride larger
      # than the target output_stride.
      current_stride = 1
    
      # The atrous convolution rate parameter.
      rate = 1
    
      for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
          for i, unit in enumerate(block.args):
            if output_stride is not None and current_stride > output_stride:
              raise ValueError('The target output_stride cannot be reached.')
    
            with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
              unit_depth, unit_depth_bottleneck, unit_stride = unit
    
              # If we have reached the target output_stride, then we need to employ
              # atrous convolution with stride=1 and multiply the atrous rate by the
              # current unit's stride for use in subsequent layers.
              if output_stride is not None and current_stride == output_stride:
                net = block.unit_fn(net,
                                    depth=unit_depth,
                                    depth_bottleneck=unit_depth_bottleneck,
                                    stride=1,
                                    rate=rate)
                rate *= unit_stride
    
              else:
                net = block.unit_fn(net,
                                    depth=unit_depth,
                                    depth_bottleneck=unit_depth_bottleneck,
                                    stride=unit_stride,
                                    rate=1)
                current_stride *= unit_stride
          net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)
    
      if output_stride is not None and current_stride != output_stride:
        raise ValueError('The target output_stride cannot be reached.')
    
      return net


def resnet_arg_scope(weight_decay=arg_parsing.WEIGHT_DECAY,
                     batch_norm_decay=arg_parsing.MOVING_AVERAGE_DECAY,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
      """Defines the default ResNet arg scope.
      TODO(gpapan): The batch-normalization related default values above are
        appropriate for use in conjunction with the reference ResNet models
        released at https://github.com/KaimingHe/deep-residual-networks. When
        training ResNets from scratch, they might need to be tuned.
      Args:
        weight_decay: The weight decay to use for regularizing the model.
        batch_norm_decay: The moving average decay when estimating layer activation
          statistics in batch normalization.
        batch_norm_epsilon: Small constant to prevent division by zero when
          normalizing activations by their variance in batch normalization.
        batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
          activations in the batch normalization layer.
      Returns:
        An `arg_scope` to use for the resnet models.
      """
      batch_norm_params = {
          'decay': batch_norm_decay,
          'epsilon': batch_norm_epsilon,
          'scale': batch_norm_scale,
          'updates_collections': tf.GraphKeys.UPDATE_OPS,
      }
    
      with slim.arg_scope(
          [slim.conv2d],
          weights_regularizer=slim.l2_regularizer(weight_decay),
          weights_initializer=slim.variance_scaling_initializer(),
          activation_fn=tf.nn.relu,
          normalizer_fn=slim.batch_norm,
          normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
          # The following implies padding='SAME' for pool1, which makes feature
          # alignment easier for dense prediction tasks. This is also used in
          # https://github.com/facebook/fb.resnet.torch. However the accompanying
          # code of 'Deep Residual Learning for Image Recognition' uses
          # padding='VALID' for pool1. You can switch to that choice by setting
          # slim.arg_scope([slim.max_pool2d], padding='VALID').
          with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
            return arg_sc


@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, rate=1,
               outputs_collections=None, scope=None):
      """Bottleneck residual unit variant with BN before convolutions.
      This is the full preactivation residual unit variant proposed in [2]. See
      Fig. 1(b) of [2] for its definition. Note that we use here the bottleneck
      variant which has an extra bottleneck layer.
      When putting together two consecutive ResNet blocks that use this unit, one
      should use stride = 2 in the last unit of the first block.
      Args:
        inputs: A tensor of size [batch, height, width, channels].
        depth: The depth of the ResNet unit output.
        depth_bottleneck: The depth of the bottleneck layers.
        stride: The ResNet unit's stride. Determines the amount of downsampling of
          the units output compared to its input.
        rate: An integer, rate for atrous convolution.
        outputs_collections: Collection to add the ResNet unit output.
        scope: Optional variable_scope.
      Returns:
        The ResNet unit's output.
      """
      with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
        if depth == depth_in:
          shortcut = subsample(inputs, stride, 'shortcut')
        else:
          shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                                 normalizer_fn=None, activation_fn=None,
                                 scope='shortcut')
    
        residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1,
                               scope='conv1')
        residual = conv2d_same(residual, depth_bottleneck, 3, stride,
                                            rate=rate, scope='conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                               normalizer_fn=None, activation_fn=None,
                               scope='conv3')
    
        output = shortcut + residual
    
        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.original_name_scope,
                                                output)


def resnet_v2(inputs,
              blocks,
              output_stride=None,
              include_root_block=True,
              reuse=None,
              scope=None):
  """Generator for v2 (preactivation) ResNet models.
  This function generates a family of ResNet v2 models. See the resnet_v2_*()
  methods for specific model instantiations, obtained by selecting different
  block instantiations that produce ResNets of various depths.
  Training for image classification on Imagenet is usually done with [224, 224]
  inputs, resulting in [7, 7] feature maps at the output of the last ResNet
  block for the ResNets defined in [1] that have nominal stride equal to 32.
  However, for dense prediction tasks we advise that one uses inputs with
  spatial dimensions that are multiples of 32 plus 1, e.g., [321, 321]. In
  this case the feature maps at the ResNet output will have spatial shape
  [(height - 1) / output_stride + 1, (width - 1) / output_stride + 1]
  and corners exactly aligned with the input image corners, which greatly
  facilitates alignment of the features to the image. Using as input [225, 225]
  images results in [8, 8] feature maps at the output of the last ResNet block.
  For dense prediction tasks, the ResNet needs to run in fully-convolutional
  (FCN) mode and global_pool needs to be set to False. The ResNets in [1, 2] all
  have nominal stride equal to 32 and a good choice in FCN mode is to use
  output_stride=16 in order to increase the density of the computed features at
  small computational and memory overhead, cf. http://arxiv.org/abs/1606.00915.
  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    blocks: A list of length equal to the number of ResNet blocks. Each element
      is a resnet_utils.Block object describing the units in the block.
    output_stride: If None, then the output will be computed at the nominal
      network stride. If output_stride is not None, it specifies the requested
      ratio of input to output spatial resolution.
    include_root_block: If True, include the initial convolution followed by
      max-pooling, if False excludes it. If excluded, `inputs` should be the
      results of an activation-less convolution.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.
  Returns:
    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
      If global_pool is False, then height_out and width_out are reduced by a
      factor of output_stride compared to the respective height_in and width_in,
      else both height_out and width_out equal one. If num_classes is None, then
      net is the output of the last ResNet block, potentially after global
      average pooling. If num_classes is not None, net contains the pre-softmax
      activations.
    end_points: A dictionary from components of the network to the corresponding
      activation.
  Raises:
    ValueError: If the target output_stride is not valid.
  """
  with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=tf.AUTO_REUSE) as sc:
    end_points_collection = sc.name + '_end_points'
    with slim.arg_scope([slim.conv2d, bottleneck,
                         stack_blocks_dense],
                        outputs_collections=end_points_collection):
      with slim.arg_scope([slim.batch_norm], is_training=is_training):
        net = inputs
        if include_root_block:
          if output_stride is not None:
            if output_stride % 4 != 0:
              raise ValueError('The output_stride needs to be a multiple of 4.')
            output_stride /= 4
          # We do not include batch normalization or activation functions in
          # conv1 because the first ResNet unit will perform these. Cf.
          # Appendix of [2].
          with slim.arg_scope([slim.conv2d],
                              activation_fn=None, normalizer_fn=None):
            net = conv2d_same(net, 64, 7, stride=2, scope='conv1')
          net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
        net = stack_blocks_dense(net, blocks, output_stride)
        # This is needed because the pre-activation variant does not have batch
        # normalization or activation functions in the residual unit output. See
        # Appendix of [2].
        net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
        if global_pool:
          # Global average pooling.
          net = tf.reduce_mean(net, [1, 2], name='pool5', keepdims=True)
        if num_classes is not None:
          net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                            normalizer_fn=None, scope='logits')
          net = tf.squeeze(net, [1,2])
        # Convert end_points_collection into a dictionary of end_points.
#        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
#        if num_classes is not None:
#          end_points['predictions'] = slim.softmax(net, scope='predictions')
        return net


def resnet_v2_50(inputs,
                 output_stride=None,
                 reuse=None,
                 scope='resnet_v2_50'):
  """ResNet-50 model of [1]. See resnet_v2() for arg and return description."""
  blocks = [
      Block(
          'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
      Block(
          'block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
      Block(
          'block3', bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
      Block(
          'block4', bottleneck, [(2048, 512, 1)] * 3)]
  return resnet_v2(inputs, blocks,output_stride=output_stride,
                   include_root_block=True, reuse=reuse, scope=scope)


def resnet_v2_101(inputs,
                  output_stride=None,
                  reuse=None,
                  scope='resnet_v2_101'):
  """ResNet-101 model of [1]. See resnet_v2() for arg and return description."""
  blocks = [
      Block(
          'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
      Block(
          'block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
      Block(
          'block3', bottleneck, [(1024, 256, 1)] * 22 + [(1024, 256, 2)]),
      Block(
          'block4', bottleneck, [(2048, 512, 1)] * 3)]
  return resnet_v2(inputs, blocks,output_stride=output_stride,
                   include_root_block=True, reuse=reuse, scope=scope)


def resnet_v2_152(inputs,
                  output_stride=None,
                  reuse=None,
                  scope='resnet_v2_152'):
  """ResNet-152 model of [1]. See resnet_v2() for arg and return description."""
  blocks = [
      Block(
          'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
      Block(
          'block2', bottleneck, [(512, 128, 1)] * 7 + [(512, 128, 2)]),
      Block(
          'block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
      Block(
          'block4', bottleneck, [(2048, 512, 1)] * 3)]
  return resnet_v2(inputs, blocks,output_stride=output_stride,
                   include_root_block=True, reuse=reuse, scope=scope)


def resnet_v2_200(inputs,
                  output_stride=None,
                  reuse=None,
                  scope='resnet_v2_200'):
  """ResNet-200 model of [2]. See resnet_v2() for arg and return description."""
  blocks = [
      Block(
          'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
      Block(
          'block2', bottleneck, [(512, 128, 1)] * 23 + [(512, 128, 2)]),
      Block(
          'block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
      Block(
          'block4', bottleneck, [(2048, 512, 1)] * 3)]
  return resnet_v2(inputs, blocks,output_stride=output_stride,
                   include_root_block=True, reuse=reuse, scope=scope)
