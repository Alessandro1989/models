# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the CIFAR-10 network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

import svhn_readInput
FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_boolean('use_fp16', False,
                            "Train the model using fp16.")
IMAGE_SIZE = svhn_readInput.IMAGE_SIZE
NUM_CLASSES = svhn_readInput.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = svhn_readInput.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = svhn_readInput.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
BATCH_SIZE = svhn_readInput.BATCH_SIZE

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.

#Learning rate:
NUM_EPOCHS_PER_DECAY = 12 #14.0      # 105 Epochs after which learning rate decays (350).
LEARNING_RATE_DECAY_FACTOR = 0.05  # 0.1 Learning rate decay factor. (before 0.1)
INITIAL_LEARNING_RATE = 0.106#0.102  #0.12    # Initial learning rate. (before was 0.1)
STAIRCASE = False #se è a true decrementa a intervalli discreti... (before was true)
#for do tests (seems not wokring:
#NUM_EPOCHS_PER_DECAY = 35.0      # Epochs after which learning rate decays (350).
#LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor. (before 0.1)
#INITIAL_LEARNING_RATE = 0.12  #0.12    # Initial learning rate. (before was 0.1)
#STAIRCASE = False #se è a true decrementa a intervalli discreti... (before was true)

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

#DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
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
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32

  #try l1 regulazire instead of l2
  #l1_regularizer = tf.contrib.layers.l1_regularizer(
  #    scale=0.005, scope=None
  #)
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))

  #list = [element.item() for element in var.flatten()]

  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    #regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, var)
    #weight_decay = tf.multiply(regularization_penalty, wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def inference(images):
  """Build the Svhn Model

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  with tf.name_scope('network'):
      with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             #shape=[5, 5, 3, 64],
                                             shape=[6, 6, 3, 90],
                                            # stddev=5e-2,
                                             stddev=5e-2,
                                             wd=None)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        #biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        biases = _variable_on_cpu('biases', [90], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv1)

        # pool1
        #pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
        #                      padding='SAME', name='pool1')
        # norm1
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool1')

        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')


        # conv2
      with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             #shape=[5, 5, 64, 64],
                                             shape=[6, 6, 90, 90],
                                             stddev=5e-2,
                                             wd=None)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        #prima era 64
        biases = _variable_on_cpu('biases', [90], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv2)

        pool2 = tf.nn.avg_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
        # norm2
        norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                        name='norm2')
      # pool2
      #pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME', name='pool2')
      #pool2 = tf.nn.avg_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
      #pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool2')


      # conv3
      with tf.variable_scope('conv3') as scope:
          kernel = _variable_with_weight_decay('weights',
                                               # shape=[5, 5, 64, 64],
                                               shape=[6, 6, 90, 90],
                                               stddev=5e-2,
                                               wd=None)
          conv = tf.nn.conv2d(norm2, kernel, [1, 1, 1, 1], padding='SAME')
          # prima era 64
          biases = _variable_on_cpu('biases', [90], tf.constant_initializer(0.1))
          pre_activation = tf.nn.bias_add(conv, biases)
          conv3 = tf.nn.relu(pre_activation, name=scope.name)
          _activation_summary(conv3)

          # pool3
          # pool3 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME', name='pool2')
          pool3 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
           ############################end##############################

          # norm3
          norm3 = tf.nn.lrn(pool3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                            name='norm2')

      """
      # conv4
      with tf.variable_scope('conv4') as scope:
          kernel = _variable_with_weight_decay('weights',
                                               # shape=[5, 5, 64, 64],
                                               shape=[6, 6, 90, 90],
                                               stddev=5e-2,
                                               wd=None)
          conv = tf.nn.conv2d(norm3, kernel, [1, 1, 1, 1], padding='SAME')
          # prima era 64
          biases = _variable_on_cpu('biases', [90], tf.constant_initializer(0.1))
          pre_activation = tf.nn.bias_add(conv, biases)
          conv4 = tf.nn.relu(pre_activation, name=scope.name)
          _activation_summary(conv4)

          # pool3
          pool4 = tf.nn.max_pool(conv4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
           ############################end##############################

          # norm3
          norm4 = tf.nn.lrn(pool4, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                            name='norm4')

      #''''
      # conv5
      with tf.variable_scope('conv5') as scope:
          kernel = _variable_with_weight_decay('weights',
                                               # shape=[5, 5, 64, 64],
                                               shape=[6, 6, 90, 90],
                                               stddev=5e-2,
                                               wd=None)
          conv = tf.nn.conv2d(norm4, kernel, [1, 1, 1, 1], padding='SAME')
          # prima era 64
          biases = _variable_on_cpu('biases', [90], tf.constant_initializer(0.1))
          pre_activation = tf.nn.bias_add(conv, biases)
          conv5 = tf.nn.relu(pre_activation, name=scope.name)
          _activation_summary(conv5)
    
      # pool5
      pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
      ############################end##############################
    
      # norm5
      norm5 = tf.nn.lrn(pool5, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                        name='norm4')
      """

      #lamdaRegularization = 0.025
      lamdaRegularization = 0.025
      #0.3
      # local3
      with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        #reshape = tf.reshape(norm4, [images.get_shape().as_list()[0], -1])
        reshape = tf.reshape(norm3, [images.get_shape().as_list()[0], -1])
        dim = reshape.get_shape()[1].value
        #weights = _variable_with_weight_decay('weights', shape=[dim, 384],
        #                                      stddev=0.04, wd=0.004)
        weights = _variable_with_weight_decay('weights', shape=[dim, 700],
                                              stddev=0.04, wd=lamdaRegularization)
        biases = _variable_on_cpu('biases', [700], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        _activation_summary(local3)


      # local4
      with tf.variable_scope('local4') as scope:
        #weights = _variable_with_weight_decay('weights', shape=[384, 192],
        #                                      stddev=0.04, wd=0.004)

       # weights = _variable_with_weight_decay('weights', shape=[384, 192],
        #                                      stddev=0.04, wd=lamdaRegularization)
        weights = _variable_with_weight_decay('weights', shape=[700, 700],
                                            stddev=0.04, wd=lamdaRegularization)
        biases = _variable_on_cpu('biases', [700], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        _activation_summary(local4)

      # linear layer(WX + b),
      # We don't apply softmax here because
      # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
      # and performs the softmax internally for efficiency.
      with tf.variable_scope('softmax_linear') as scope:
       # weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
       #                                       stddev=1/192.0, wd=None)
       weights = _variable_with_weight_decay('weights', [700, NUM_CLASSES],
                             stddev=1/700.0, wd=None)
       biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                  tf.constant_initializer(0.0))
       #softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
       softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
       _activation_summary(softmax_linear)

       return softmax_linear


def loss(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train SVHN model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.

  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  STAIRCASE) #before was true
  #op_printlabel = tf.Print(pngName, [pngName], "tensorLabel")
  op_printlearningrate = tf.Print(lr, [lr,global_step,decay_steps], "learning rate: ; global_step: ; decay_steps: ")

  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([op_printlearningrate]):
      with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  with tf.control_dependencies([apply_gradient_op]):
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

  return variables_averages_op

