#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 10:25:31 2018

@author: hans
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
from datetime import datetime

import tensorflow as tf
from tensorflow.python import debug as tfdbg
import arg_parsing
import dataset
from net import squeezenet
from net import mobilenet
from net import mobilenetv2
from net import resnet

FLAGS = arg_parsing.parser.parse_args()

def _logits(images):
    if arg_parsing.NET == 'squeezenet':
        logits = squeezenet.inference(images)
    elif arg_parsing.NET == 'mobilenet':
        logits = mobilenet.inference(images)
    elif arg_parsing.NET == 'mobilenetv2':
        logits = mobilenetv2.inference(images)
    elif arg_parsing.NET == 'resnet':
        if arg_parsing.RESNET_LAYER_NUM==50:
            logits = resnet.resnet_v2_50(images)
        elif arg_parsing.RESNET_LAYER_NUM==101:
            logits = resnet.resnet_v2_101(images)
        elif arg_parsing.RESNET_LAYER_NUM==152:
            logits = resnet.resnet_v2_152(images)
        elif arg_parsing.RESNET_LAYER_NUM==200:
            logits = resnet.resnet_v2_200(images)
    return logits
    
def _loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def _optimization(total_loss, global_step):
    num_batches_per_epoch = arg_parsing.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / arg_parsing.BATCH_SIZE
    decay_steps = int(num_batches_per_epoch * arg_parsing.NUM_EPOCHS_PER_DECAY)
    lr = tf.train.exponential_decay(arg_parsing.INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    arg_parsing.LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    with tf.name_scope("lr"):
        tf.summary.scalar('lr', lr)
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    variable_averages = tf.train.ExponentialMovingAverage(
                        arg_parsing.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')
    return train_op

def val(train_loss):
    with tf.device('/cpu:0'):
        val_images, val_labels = dataset.process_inputs("val")
    val_logits = _logits(val_images)
    val_loss = _loss(val_logits, val_labels)
    
    val_acc = tf.nn.in_top_k(val_logits,val_labels,1)
    val_acc_sum = tf.cast(val_acc,tf.float32)
    val_acc_sum = tf.reduce_mean(val_acc_sum)
    
    with tf.name_scope("loss"):
        tf.summary.scalar('train_loss', train_loss)
        tf.summary.scalar('val_loss', val_loss)
    return val_acc_sum

def train():
    global_step = tf.Variable(0, dtype=tf.int32, name='global_step', trainable=False)
    with tf.device('/cpu:0'):
        images, labels = dataset.process_inputs("training")
    logits = _logits(images)
    loss = _loss(logits, labels)  
    train_op = _optimization(loss, global_step)
    with tf.name_scope("global_step"):
        tf.summary.scalar('global_step', global_step)
    val_step = int(math.ceil(arg_parsing.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL/arg_parsing.BATCH_SIZE))
    val_acc_sum = val(loss)
    all_hooks=[tf.train.NanTensorHook(loss)]
    if arg_parsing.DEBUG:
        all_hooks.append(tfdbg.LocalCLIDebugHook(ui_type='curses'))
    config = tf.ConfigProto(log_device_placement=arg_parsing.LOG_DEVICE_PLACEMENT)
    config.gpu_options.allow_growth=True
    with tf.train.MonitoredTrainingSession(
            checkpoint_dir=arg_parsing.MODEL_DIR,
            hooks=all_hooks,
            config=config,
            save_summaries_steps=100,
            save_summaries_secs=None,
            log_step_count_steps=None) as sess:
        total_loss = 0
        start_time = time.time()
        for i in range(1, arg_parsing.MAX_STEPS+1):
            _,loss_value = sess.run([train_op,loss])
            total_loss += loss_value
            if i % arg_parsing.LOG_FREQUENCY == 0:
                current_time = time.time()
                duration = current_time - start_time
                eg_per_sec = arg_parsing.LOG_FREQUENCY * arg_parsing.BATCH_SIZE / duration
                sec_per_batch = float(duration / arg_parsing.LOG_FREQUENCY)
                avg_loss = total_loss/i
                print('%s: training step %d cur loss = %.4f avg loss = %.4f (%.1f images/sec, %.3f sec/batch)'
                      % (datetime.now(), i, loss_value, avg_loss, eg_per_sec, sec_per_batch))
                start_time = time.time()
            if i % arg_parsing.STEPS_TO_VAL == 0:
                total_val_accu=0
                for j in range(val_step):
                    total_val_accu+=sess.run(val_acc_sum)
                print('%s: validation total accuracy = %.4f (%.3f sec %d batches)'
                      % (datetime.now(), total_val_accu/float(val_step), float(time.time()-start_time), val_step))
                start_time = time.time()
                
def train_dis_():
    ps_hosts = arg_parsing.PS_HOSTS.split(",")
    worker_hosts = arg_parsing.WORKER_HOSTS.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)
    if FLAGS.job_name == "ps":
        server.join()
    if FLAGS.job_name == "worker":
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" 
                                                      %FLAGS.task_index,cluster=cluster)):
            global_step = tf.Variable(0, dtype=tf.int32, name='global_step', trainable=False)
            with tf.device('/cpu:0'):
                images, labels = dataset.process_inputs("training")
            logits = _logits(images)
            loss = _loss(logits, labels)  
            train_op = _optimization(loss, global_step)
            with tf.name_scope("global_step"):
                tf.summary.scalar('global_step', global_step)

            val_step = int(math.ceil(arg_parsing.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL/arg_parsing.BATCH_SIZE))
            val_acc_sum = val(loss)

            all_hooks=[tf.train.NanTensorHook(loss)]
            if arg_parsing.DEBUG:
                all_hooks.append(tfdbg.LocalCLIDebugHook(ui_type='curses'))
            config = tf.ConfigProto(log_device_placement=arg_parsing.LOG_DEVICE_PLACEMENT)
            config.gpu_options.allow_growth=True
            with tf.train.MonitoredTrainingSession(
                    master=server.target,
                    is_chief=(FLAGS.task_index == 0),
                    checkpoint_dir=arg_parsing.MODEL_DIR,
                    hooks=all_hooks,
                    config=config,
                    save_summaries_steps=100,
                    save_summaries_secs=None,
                    log_step_count_steps=None) as sess:
                total_loss = 0
                start_time = time.time()
                for i in range(1, arg_parsing.MAX_STEPS+1):
                    _,loss_value = sess.run([train_op,loss])
                    total_loss += loss_value
                    if i % arg_parsing.LOG_FREQUENCY == 0:
                        current_time = time.time()
                        duration = current_time - start_time
                        eg_per_sec = arg_parsing.LOG_FREQUENCY * arg_parsing.BATCH_SIZE / duration
                        sec_per_batch = float(duration / arg_parsing.LOG_FREQUENCY)
                        avg_loss = total_loss/i
                        print('%s: training step %d cur loss = %.4f avg loss = %.4f (%.1f images/sec, %.3f sec/batch)'
                              % (datetime.now(), i, loss_value, avg_loss, eg_per_sec, sec_per_batch))
                        start_time = time.time()
                    if i % arg_parsing.STEPS_TO_VAL == 0:
                        total_val_accu=0
                        for j in range(val_step):
                            total_val_accu+=sess.run(val_acc_sum)
                        print('%s: validation total accuracy = %.4f (%.3f sec %d batches)'
                              % (datetime.now(), total_val_accu/float(val_step), float(time.time()-start_time), val_step))
                        start_time = time.time()
