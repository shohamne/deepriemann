from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/neta/Downloads/mnist/", one_hot=True)

import tensorflow as tf

from pymanopt import Problem
from pymanopt.solvers import SGD
from pymanopt.manifolds import FixedRankEmbedded, Euclidean, Product, Stiefel


#parameters
k = 5
weight_decay = 0.1
learning_rate_starter = 0.01
learning_rate_decay_steps =10000
learning_rate_decay_rate = 0.5

use_parameterization = False


with tf.device("/cpu:0"):
    #sess = tf.InteractiveSession()
    x = tf.placeholder("float", [None, 784], name="x-input")

    b = tf.Variable(tf.zeros([1, 10]), name="bias")
    b_hist = tf.histogram_summary("bias-hist", b)


    A = tf.Variable(tf.random_uniform([784,k]), name="A-weights")

    B = tf.Variable(tf.random_uniform([k,10]), name="B-weights")

    if use_parameterization:
        W = tf.matmul(A, B, name="W-weights")
    else:
        M = tf.Variable(tf.random_uniform([k]), name="M-weights")
        W = tf.matmul(tf.matmul(A, tf.diag(M)), B, name="W-weights")

    eW = tf.Variable(tf.random_uniform([784, 10]), name="eW-weights")

    A_hist = tf.histogram_summary("A-hist", A)
    if not use_parameterization:
        M_hist = tf.histogram_summary("M-hist", M)
    B_hist = tf.histogram_summary("B-hist", B)
    W_hist = tf.histogram_summary("W-hist", W)

#with tf.device("/gpu:0"):

    #with tf.name_scope("xA") as scope:
    #  xA = tf.nn.softmax(tf.matmul(x,A))
    #with tf.name_scope("xAM") as scope:
    #  xAM = tf.nn.softmax(tf.matmul(xA,tf.diag(M,'diag_M')))
    #with tf.name_scope("xAMB_b") as scope:
    #  lin_y = tf.matmul(xAM,B) + b
    with tf.name_scope("xW_b") as scope:
        lin_y = tf.matmul(x, W) + b
    with tf.name_scope("e_xW_b") as scope:
        e_lin_y = tf.matmul(x,eW) + b
    with tf.name_scope("logit") as scope:
      y = tf.nn.softmax(lin_y)
    with tf.name_scope("e_logit") as scope:
      e_y = tf.nn.softmax(e_lin_y)
#with tf.device("/cpu:0"):
    y_hist = tf.histogram_summary("y", y)

#with tf.device("/cpu:0"):
    # Define loss and optimizer
    with tf.name_scope("regulization") as scope:
        regulize_A = tf.nn.l2_loss(A)
        regulize_B = tf.nn.l2_loss(B)

        rgA_summ = tf.scalar_summary("regulize A", regulize_A)
        rgB_summ = tf.scalar_summary("regulize B", regulize_B)
        if not use_parameterization:
            regulize_M = tf.nn.l2_loss(M)
            rgM_summ = tf.scalar_summary("regulize M", regulize_M)

        regulize_W = tf.nn.l2_loss(W)
        rgW_summ = tf.scalar_summary("regulize W", regulize_W)

        regulize_eW = tf.nn.l2_loss(eW)
        #e_rgW_summ = tf.scalar_summary("regulize eW", regulize_eW)

        y_ = tf.placeholder("float", [None,10], name="y-input" )

    with tf.name_scope("xent") as scope:
      cross_entropy = -tf.reduce_sum(y_*tf.log(y))
      e_cross_entropy = -tf.reduce_sum(y_ * tf.log(e_y))
      #cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(lin_y,y_))
      ce_summ = tf.scalar_summary("cross entropy", cross_entropy)


      loss = cross_entropy + (weight_decay*regulize_W)
      e_loss = e_cross_entropy + (weight_decay * regulize_eW)

      loss_sum = tf.scalar_summary("loss", loss)
      

    
    with tf.name_scope("test") as scope:
      correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
      accuracy_summary = tf.scalar_summary("accuracy", accuracy)

    #with tf.name_scope("e_test") as scope:
        #e_correct_prediction = tf.equal(tf.argmax(e_y, 1), tf.argmax(y_, 1))
        #e_accuracy = tf.reduce_mean(tf.cast(e_correct_prediction, "float"))
        #e_accuracy_summary = tf.scalar_summary("e_accuracy", e_accuracy)

    summaries = tf.merge_all_summaries()



    manifold_b = Euclidean(1,10)
    if use_parameterization:
        manifold_A = Euclidean(784, k)
        manifold_B = Euclidean(k, 10)
        manifold_W = Product([manifold_A,manifold_B])
        arg = [A, B, b]
    else:
        manifold_W = FixedRankEmbedded(784, 10, k)
        #manifold_W = Product([Stiefel(784,k), Euclidean(k), Stiefel(k,10)])
        arg = [A, M, B, b]

    manifold = Product([manifold_W, manifold_b])

    e_manifold_W = Euclidean(784, 10)
    e_arg = [eW, b]

    e_manifold = Product([e_manifold_W, manifold_b])

    problem = Problem(manifold=manifold, cost=loss, accuracy=accuracy, summary=summaries, arg=arg, data=[x,y_], verbosity=1)
    e_problem = Problem(manifold=e_manifold, cost=e_loss, accuracy=accuracy, summary=summaries, arg=e_arg, data=[x, y_],
                      verbosity=1)

    solver = SGD(maxiter=10000000,logverbosity=10)

    solver.solve(problem, e_problem, mnist, 10, learning_rate_starter,
                                learning_rate_decay_steps,learning_rate_decay_rate)

