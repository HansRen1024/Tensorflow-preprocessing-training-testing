#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 10:25:31 2018

@author: hans
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from datetime import datetime

import tensorflow as tf
from tensorflow.python import debug as tfdbg
import arg_parsing
import dataset
from net import network
from net import squeezenet
from net import mobilenet
from net import mobilenetv2
from net import resnet

FLAGS = arg_parsing.parser.parse_args()
config = tf.ConfigProto(log_device_placement=arg_parsing.LOG_DEVICE_PLACEMENT)
config.gpu_options.allow_growth=True

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

class _LoggerHook(tf.train.SessionRunHook):
    def begin(self):
        self._step = -1
        self._start_time = time.time()
        self._total_loss = 0

    def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)

    def after_run(self, run_context, run_values):
        loss_value = run_values.results
        self._total_loss += loss_value
        if self._step % arg_parsing.LOG_FREQUENCY == 0:
            current_time = time.time()
            duration = current_time - self._start_time
            self._start_time = current_time
            if self._step==0:
                avg_loss = loss_value
            else:
                avg_loss = self._total_loss/self._step
            eg_per_sec = arg_parsing.LOG_FREQUENCY * arg_parsing.BATCH_SIZE / duration
            sec_per_batch = float(duration / arg_parsing.LOG_FREQUENCY)
            print('%s: training step %d, current loss = %.4f, avg loss = %.4f (%.1f images/sec; %.3f sec/batch)'
                  % (datetime.now(), self._step, loss_value, avg_loss, eg_per_sec, sec_per_batch))
#        if FLAGS.task_index==0 and self._step % arg_parsing.STEPS_TO_VAL == 0: #慎用，还没弄好
#            test.test('val')

def train():
    global loss
    global_step = tf.Variable(0, dtype=tf.int32, name='global_step', trainable=False)
    with tf.device('/cpu:0'):
        images, labels = dataset.process_inputs("training")
    logits = _logits(images)
    loss = _loss(logits, labels)
    tf.summary.scalar('loss', loss)
    acc = tf.nn.in_top_k(logits,labels,1)
    acc = tf.cast(acc,tf.float32)
    acc = tf.reduce_mean(acc)
    tf.summary.scalar('accuracy', acc)
    train_op = _optimization(loss, global_step)
    
    if arg_parsing.DEBUG:
        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=arg_parsing.MODEL_DIR,
                hooks=[tf.train.StopAtStepHook(last_step=arg_parsing.MAX_STEPS),
                       tf.train.NanTensorHook(loss), # Monitors the loss tensor and stops training if loss is NaN.
                       tfdbg.LocalCLIDebugHook(ui_type='curses'), # Command-line-interface debugger hook.
#                       tfdbg.TensorBoardDebugHook(grpc_debug_server_addresses="localhost:6000"), # can be used with TensorBoard Debugger Plugin.
                       _LoggerHook()],
                config=config) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)
    else:
        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=arg_parsing.MODEL_DIR,
                hooks=[tf.train.StopAtStepHook(last_step=arg_parsing.MAX_STEPS),
                       tf.train.NanTensorHook(loss), # Monitors the loss tensor and stops training if loss is NaN.
                       _LoggerHook()],
                config=config) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)

def train_dis_():
    global loss
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
            images, labels = dataset.process_inputs("training")
            logits = _logits(images)
            loss = _loss(logits, labels)
            tf.summary.scalar('loss', loss)
            acc = tf.nn.in_top_k(logits,labels,1)
            acc = tf.cast(acc,tf.float32)
            acc = tf.reduce_mean(acc)
            tf.summary.scalar('accuracy', acc)
            train_op = _optimization(loss, global_step)
            if arg_parsing.DEBUG:
                with tf.train.MonitoredTrainingSession(
                        master=server.target,
                        is_chief=(FLAGS.task_index == 0),
                        checkpoint_dir=arg_parsing.MODEL_DIR,
                        hooks=[tf.train.StopAtStepHook(last_step=arg_parsing.MAX_STEPS),
                               tf.train.NanTensorHook(loss), # Monitors the loss tensor and stops training if loss is NaN.
                               tfdbg.LocalCLIDebugHook(ui_type='curses'), # Command-line-interface debugger hook.
    #                           tfdbg.TensorBoardDebugHook(grpc_debug_server_addresses="localhost:6000"), # can be used with TensorBoard Debugger Plugin.
                               _LoggerHook()],
                        config=config) as mon_sess:
                    while not mon_sess.should_stop():
                        mon_sess.run(train_op)
            else:
                with tf.train.MonitoredTrainingSession(
                        master=server.target,
                        is_chief=(FLAGS.task_index == 0),
                        checkpoint_dir=arg_parsing.MODEL_DIR,
                        hooks=[tf.train.StopAtStepHook(last_step=arg_parsing.MAX_STEPS),
                               tf.train.NanTensorHook(loss), # Monitors the loss tensor and stops training if loss is NaN.
                               _LoggerHook()],
                        config=config) as mon_sess:
                    while not mon_sess.should_stop():
                        mon_sess.run(train_op)
            
def train_dis():
    ps_hosts = arg_parsing.PS_HOSTS.split(",")
    worker_hosts = arg_parsing.WORKER_HOSTS.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster,job_name=FLAGS.job_name,task_index=FLAGS.task_index)
    issync = FLAGS.issync 
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index,cluster=cluster)):
            global_step = tf.Variable(0, dtype=tf.int32, name='global_step', trainable=False)
#            global_step = tf.train.get_or_create_global_step()
            input = tf.placeholder("float")
            label = tf.placeholder("float")
            images, labels = dataset.process_inputs("training")
            logits = network.inference(images, True)
            loss = _loss(logits, labels)
            train_prediction = tf.nn.softmax(logits)
            num_batches_per_epoch = arg_parsing.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / arg_parsing.BATCH_SIZE
            decay_steps = int(num_batches_per_epoch * arg_parsing.NUM_EPOCHS_PER_DECAY)
            lr = tf.train.exponential_decay(arg_parsing.INITIAL_LEARNING_RATE,
                                            global_step,
                                            decay_steps,
                                            arg_parsing.LEARNING_RATE_DECAY_FACTOR,
                                            staircase=True)
            loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
            losses = tf.get_collection('losses')
            loss_averages_op = loss_averages.apply(losses + [loss])
            with tf.control_dependencies([loss_averages_op]):
                optimizer = tf.train.GradientDescentOptimizer(lr)
                grads_and_vars = optimizer.compute_gradients(loss)
            if issync == 1:
                #同步模式计算更新梯度
                rep_op = tf.train.SyncReplicasOptimizer(optimizer,
                                                        replicas_to_aggregate=len(worker_hosts),
                                                        total_num_replicas=len(worker_hosts),
                                                        use_locking=True)
                apply_gradient_op = rep_op.apply_gradients(grads_and_vars, global_step=global_step)
                variable_averages = tf.train.ExponentialMovingAverage(arg_parsing.MOVING_AVERAGE_DECAY, global_step)
                variables_averages_op = variable_averages.apply(tf.trainable_variables())
                with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
                    train_op = tf.no_op(name='train')
                init_token_op = rep_op.get_init_tokens_op()
                chief_queue_runner = rep_op.get_chief_queue_runner()
            else:
                #异步模式计算更新梯度
                apply_gradient_op = optimizer.apply_gradients(grads_and_vars,global_step=global_step)
                variable_averages = tf.train.ExponentialMovingAverage(arg_parsing.MOVING_AVERAGE_DECAY, global_step)
                variables_averages_op = variable_averages.apply(tf.trainable_variables())
            
                with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
                    train_op = tf.no_op(name='train')
            init_op = tf.global_variables_initializer()
            saver = tf.train.Saver()
            tf.summary.scalar('loss', loss)
            summary_op = tf.summary.merge_all()
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                logdir=arg_parsing.MODEL_DIR,
                                init_op=init_op,
                                summary_op=None,
                                saver=saver,
                                global_step=global_step,
                                save_model_secs=600)
#        coord = tf.train.Coordinator()
        with sv.prepare_or_wait_for_session(server.target) as sess:
            # 如果是同步模式
            if FLAGS.task_index == 0 and issync == 1:
                sv.start_queue_runners(sess, [chief_queue_runner])
                sess.run(init_token_op)
#            if FLAGS.task_index == 0 and issync == 0:
#                coord = tf.train.Coordinator()
#                threads = sv.start_queue_runners(sess=sess, queue_runners=coord)
            step = 0
            while  step < arg_parsing.MAX_STEPS:
                _, loss_v, step = sess.run([train_op, loss, global_step], feed_dict={input:images.eval(), label:labels.eval()})
                if step % arg_parsing.LOG_FREQUENCY == 0:
                    prediction = sess.run(train_prediction)
                    print("training step: %d, loss: %f, prediction: %.4f" %(step, loss_v, prediction))
#            if FLAGS.task_index == 0 and issync == 0:
#                coord.request_stop()
#                coord.join(threads)
        sv.stop()
