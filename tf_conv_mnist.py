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

def cnn_model_fn(features, labels, mode, rank):
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
  if rank == 'full':
    dense = layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu, summaries=['norm','histogram'])
    manifold = None
    manifold_args = None
  else:
    dense, fixed_rank_manifold, fixed_rank_manifold_args = \
        layers.fixed_rank_riemannian(inputs=pool2_flat, units=1024, activation=tf.nn.relu,rank=rank, summaries=['norm','histogram'])

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=10)

  loss = None
  train_op = None

  # Calculate Loss (for both TRAIN and EVAL modes)
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

  if not rank == 'full':
      all_args = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) #TODO: don't use globals
      euclidian_args = [arg for arg in all_args if arg not in fixed_rank_manifold_args]
      euclidian_manifolds = [manifolds.Euclidean(*arg.get_shape().as_list()) for arg in euclidian_args]

      manifold_args = euclidian_args + fixed_rank_manifold_args
      manifold = manifolds.Product(euclidian_manifolds+[fixed_rank_manifold])
  else:
      manifold = None
      manifold_args = None


  # Return a ModelFnOps object
  model = model_fn_lib.ModelFnOps(
      mode=mode, predictions=predictions, loss=loss, train_op=train_op)


  return model, manifold, manifold_args


def train(rank=int(1024/2), #'full' #'full' #1024
          maxiter=5000,
          learning_rate_starter=0.01,
          learning_rate_decay_steps=10000,
          learning_rate_decay_rate=0.5,
          batch_size=100):

    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/home/neta/Downloads/mnist/")

    x = tf.placeholder("float", [None, 784], name="x-input")
    y = tf.placeholder("int32", [None], name="y-input")

    model, manifold, manifold_args = cnn_model_fn(x,y,'train',rank)

    correct_prediction = tf.equal(tf.cast(model.predictions['classes'], y.dtype), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    tf.summary.scalar("accuracy", accuracy)
    tf.summary.scalar("loss", model.loss)
    summaries = tf.summary.merge_all()

    now = datetime.now()
    logdir = path.join('/tmp/tf_beackend_logs',
                       "{}-rank={}".format(now.strftime("%Y%m%d-%H%M%S"), rank))

    if not rank=='full':
        problem = Problem(manifold=manifold, cost=model.loss, accuracy=accuracy, summary=summaries, arg=manifold_args, data=[x, y],
                          verbosity=1, logdir=logdir)
        solver = SGD(maxiter=maxiter, logverbosity=10, maxtime=1000000)
        solver.solve(problem, None, mnist, batch_size, learning_rate_starter,
                     learning_rate_decay_steps, learning_rate_decay_rate)

    else:
        sess = tf.InteractiveSession()

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate_starter, global_step,
                                                   learning_rate_decay_steps, learning_rate_decay_rate, staircase=True)
        learning_step = (tf.train.GradientDescentOptimizer(learning_rate)
                         .minimize(model.loss, global_step=global_step))

        tf.initialize_all_variables().run()

        writer = tf.summary.FileWriter(logdir, sess.graph_def)

        for i in range(maxiter):
            if i % 10 == 0:  # Record summary data, and the accuracy
                feed = {x: mnist.test.images, y: mnist.test.labels}
                result = sess.run([summaries, accuracy], feed_dict=feed)
                summary_str = result[0]
                acc = result[1]
                writer.add_summary(summary_str, i)
                print("Accuracy at step %s: %s" % (i, acc))
            else:
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                feed = {x: batch_xs, y: batch_ys}
                sess.run(learning_step, feed_dict=feed)

        print(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))




# def main_(unused_argv):
#   # Load training and eval data
#   mnist = learn.datasets.load_dataset("mnist")
#   train_data = mnist.train.images  # Returns np.array
#   train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
#   eval_data = mnist.test.images  # Returns np.array
#   eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
#
#   # Create the Estimator
#   mnist_classifier = learn.Estimator(
#       model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")
#
#   # Set up logging for predictions
#   # Log the values in the "Softmax" tensor with label "probabilities"
#   tensors_to_log = {"probabilities": "softmax_tensor"}
#   logging_hook = tf.train.LoggingTensorHook(
#       tensors=tensors_to_log, every_n_iter=50)
#
#   # Train the model
#   mnist_classifier.fit(
#       x=train_data,
#       y=train_labels,
#       batch_size=100,
#       steps=20000,
#       monitors=[logging_hook])
#
#   # Configure the accuracy metric for evaluation
#   metrics = {
#       "accuracy":
#           learn.MetricSpec(
#               metric_fn=tf.metrics.accuracy, prediction_key="classes"),
#   }
#
#   # Evaluate the model and print results
#   eval_results = mnist_classifier.evaluate(
#       x=eval_data, y=eval_labels, metrics=metrics)
#   print(eval_results)

def main(unused_argv):
    if len(unused_argv)<2:
        rank = 512
    else:
        rank = unused_argv[1]
        if rank.isdigit():
            rank = int(rank)

    train(rank=rank)

if __name__ == "__main__":
  tf.app.run()