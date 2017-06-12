#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
from os import path
import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

from pymanopt import Problem
from pymanopt.solvers import SGD
from pymanopt import manifolds

import layers

def cnn(features, labels, mode, rank, is_riemannian, dropout_rate):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features, [-1, 28, 28, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  riemannian_manifolds = []
  riemannian_args = []
  if rank == 'full':
    dense = layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)  #, summaries=['norm','histogram'])
    manifold = None
    manifold_args = None
  else:
    if is_riemannian:
        dense, fixed_rank_manifold, fixed_rank_manifold_args = \
            layers.fixed_rank_riemannian(inputs=pool2_flat, units=1024, activation=tf.nn.relu,rank=rank)
                                     #, #summaries=['norm','histogram'])
        riemannian_manifolds += [fixed_rank_manifold]
        riemannian_args += fixed_rank_manifold_args
    else:
        #regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        regularizer = None
        dense0 = layers.dense(inputs=pool2_flat, units=rank, kernel_regularizer=regularizer,summaries=['norm'])
        dense= layers.dense(inputs=dense0, units=rank, kernel_regularizer=regularizer,summaries=['norm'])



  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=dropout_rate, training=mode == learn.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=10)

  loss = None
  train_op = None

  # Calculate Loss and Accuracy (for both TRAIN and EVAL modes)
  if mode != learn.ModeKeys.INFER:
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == learn.ModeKeys.TRAIN:
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.001,
        optimizer="SGD")

  # Generate Predictions
  classes =  tf.argmax(
          input=logits, axis=1)

  predictions = {
      "classes": tf.argmax(
          input=logits, axis=1),
      "probabilities": tf.nn.softmax(
          logits, name="softmax_tensor")
  }

  accuracy = None

  # Calculate Accuracy (for both TRAIN and EVAL modes)
  if mode != learn.ModeKeys.INFER:        
    correct_prediction = tf.equal(tf.cast(predictions['classes'], labels.dtype), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"),name='accuracy')
  
  manifold = None
  manifold_args = None


  all_args = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) #TODO: don't use globals
  euclidian_args = [arg for arg in all_args if arg not in riemannian_args]
  euclidian_manifolds = [manifolds.Euclidean(*arg.get_shape().as_list()) for arg in euclidian_args]
  manifold_args = euclidian_args + riemannian_args
  manifold = manifolds.Product(euclidian_manifolds+riemannian_manifolds)

  nn = {
      'predictions': predictions,
      'loss': loss,
      'train_op': train_op,
      'accuracy': accuracy,
      'manifold': manifold,
      'manifold_args': manifold_args
  }

  return nn

def cnn_model_fn(features, labels, mode, params):
    nn = cnn(features, labels, mode, params['rank'],params['is_riemannian'])
    # Return a ModelFnOps object
    model = model_fn_lib.ModelFnOps(
        mode=mode, predictions=nn['predictions'], loss=nn['loss'], train_op=nn['train_op'])

    return model


def train(rank, #'full' #'full' #1024
          is_riemannian,
          dropout_rate,
          maxiter=10000,
          batch_size=100,
          epoch=100):

    from tensorflow.examples.tutorials.mnist import input_data

    r = rank if rank != 'full' else 1024

    q1 = ( np.log2(1024) / np.log2(r))
    q2 = 1.0#q1#**2
    learning_rate_starter = 0.01*q1
    learning_rate_end = 0.001*q2
    learning_rate_decay_steps = 10
    learning_rate_decay_rate = (learning_rate_end/learning_rate_starter) \
                               ** (float(learning_rate_decay_steps)/maxiter)


    mnist = input_data.read_data_sets("/home/neta/Downloads/mnist/")

    x = tf.placeholder("float", [None, 784], name="x-input")
    y = tf.placeholder("int32", [None], name="y-input")

    with tf.variable_scope('mnist') as scope:
        nn_train = cnn(x, y, learn.ModeKeys.TRAIN, rank, is_riemannian, dropout_rate)
    with tf.variable_scope(scope, reuse=True):
        nn_eval = cnn(x, y, learn.ModeKeys.EVAL, rank, is_riemannian, dropout_rate)

    #tf.summary.scalar("trian_loss", nn_train['loss'])

    #tf.summary.scalar("trian_accuracy", nn_train['accuracy']) 
    #train_summaries = tf.summary.merge_all()
    train_summaries = tf.summary.merge([
        tf.summary.scalar("train_loss", nn_eval['loss']),
        tf.summary.scalar("train_accuracy", nn_eval['accuracy']),
        tf.summary.scalar("train_loss_dropout", nn_train['loss']),
        tf.summary.scalar("train_accuracy_dropout", nn_train['accuracy'])
    ])
    test_summaries = tf.summary.merge([
        tf.summary.scalar("test_loss", nn_eval['loss']),
        tf.summary.scalar("test_accuracy", nn_eval['accuracy']),
        tf.summary.scalar("test_loss_dropout", nn_train['loss']),
        tf.summary.scalar("test_accuracy_dropout", nn_train['accuracy'])
    ])


    now = datetime.now()
    logdir = path.join('tf_backend_logs',
                       "{}-rank={}-is_riemannian={}=dropout_rate={}".format(now.strftime("%Y%m%d-%H%M%S"),
                                                            rank, is_riemannian, dropout_rate))

    if True: #not rank=='full':
        problem = Problem(manifold=nn_train['manifold'], cost=nn_eval['loss'], accuracy=nn_eval['accuracy'],
                          cost_dropout=nn_train['loss'], accuracy_dropout=nn_train['accuracy'],
                          train_summary=train_summaries, test_summary=test_summaries,
                          arg=nn_train['manifold_args'], data=[x, y],
                          verbosity=1, logdir=logdir)
        solver = SGD(maxiter=maxiter, logverbosity=10, maxtime=1000000)
        solver.solve(problem, mnist, batch_size, learning_rate_starter,
                     learning_rate_decay_steps, learning_rate_decay_rate, epoch)

    else:
        sess = tf.InteractiveSession()

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate_starter, global_step,
                                                   learning_rate_decay_steps, learning_rate_decay_rate, staircase=True)
        learning_step = (tf.train.GradientDescentOptimizer(learning_rate)
                         .minimize(nn_train['loss'], global_step=global_step))

        tf.initialize_all_variables().run()

        writer = tf.summary.FileWriter(logdir, sess.graph_def)

        for i in range(maxiter):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feed = {x: batch_xs, y: batch_ys}
            if i==0:
                summary_str = sess.run(train_summaries, feed_dict=feed)
            writer.add_summary(summary_str, i)
            if i % epoch == 0:  # Record summary data, and the accuracy
                feed = {x: mnist.test.images, y: mnist.test.labels}
                result = sess.run([test_summaries, nn_eval['accuracy']], feed_dict=feed)
                summary_str = result[0]
                acc = result[1]
                writer.add_summary(summary_str, i)
                print("Accuracy at step %s: %s" % (i, acc))
            result = sess.run([train_summaries, learning_step], feed_dict=feed)
            summary_str = result[0]

        print(nn_train['accuracy'].eval({x: mnist.test.images, y: mnist.test.labels}))

def main(unused_argv):
    #train_orig(); return

    if len(unused_argv)<2:
        rank = 'full'
        rank = 2
    else:
        rank = unused_argv[1]
        if rank.isdigit():
            rank = int(rank)

    if len(unused_argv)<3:
        is_riemannian = True
    else:
        is_riemannian = unused_argv[2] == 'riemannian'

    if len(unused_argv) < 4:
        dropout_rate=0.4
    else:
        dropout_rate=float(unused_argv[3])


    print ("rank: {},    is_riemannian: {},     dropout_rate: {}".format(rank, is_riemannian,dropout_rate))
    train(rank, is_riemannian, dropout_rate)

#####################################################################33
# origunal training

def train_orig():
  # Load training and eval data
  mnist = learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images  # Returns np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images  # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  # Create the Estimator
  mnist_classifier = learn.Estimator(
      model_fn=cnn_model_fn,
      params={'rank': 'full'},
      model_dir="/tmp/mnist_convnet_model")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {'loss': 'softmax_cross_entropy_loss/value:0'} #{"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  mnist_classifier.fit(
      x=train_data,
      y=train_labels,
      batch_size=100,
      steps=100,#20000,
      monitors=[logging_hook])

  # Configure the accuracy metric for evaluation
  metrics = {
      "accuracy":
          learn.MetricSpec(
              metric_fn=tf.metrics.accuracy, prediction_key="classes"),
  }

  # Evaluate the model and print results
  eval_results = mnist_classifier.evaluate(
      x=eval_data, y=eval_labels, metrics=metrics)
  print(eval_results)

if __name__ == "__main__":
  tf.app.run()

