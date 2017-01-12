# -*- coding: utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/neta/Downloads/mnist/", one_hot=True)


import tensorflow as tf

with tf.device("/cpu:0"):
    x = tf.placeholder(tf.float32, [None, 784])
    
    with tf.name_scope('hidden'):
        W1 = tf.Variable(tf.random_uniform([784, 5]))
        W2 = tf.Variable(tf.random_uniform([5, 10]))
    b = tf.Variable(tf.zeros([10]))
    y_ = tf.placeholder(tf.float32, [None, 10])
    
    y = tf.nn.softmax(tf.matmul(tf.matmul(x, W1),W2) + b)
    
    
    with tf.name_scope('loss'):
        #cross_entropy 
        #loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
        #l2
        loss = tf.reduce_mean((y-y_)**2,reduction_indices=[1])
        
    train_step = tf.train.GradientDescentOptimizer(1).minimize(loss)
    
    loss_summary = tf.summary.scalar('loss',loss)
    
    init = tf.global_variables_initializer()
    
    sess = tf.Session()
    
    merged = tf.summary.merge_all()
    
    #shutil.rmtree('/tmp/train')
    train_writer = tf.summary.FileWriter('/tmp/train', sess.graph)
    
    sess.run(init)
    
    for i in range(10):
      
      batch_xs, batch_ys = mnist.train.next_batch(1000)
      summary, _  = sess.run([merged, train_step], feed_dict={x: batch_xs, y_: batch_ys})
      train_writer.add_summary(summary, i)
      
      #sess.run(merged)
      
      correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))






