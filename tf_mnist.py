from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/neta/Downloads/mnist/", one_hot=True)
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
import tensorflow as tf

#parameters
k = 5
weight_decay = 0.1
learning_rate_starter = 0.01
learning_rate_decay_steps =10000
learning_rate_decay_rate = 0.5


with tf.device("/cpu:0"):
    sess = tf.InteractiveSession()
    
    # Create the model
    x = tf.placeholder("float", [None, 784], name="x-input")
    A = tf.Variable(tf.random_uniform([784,k]), name="A-weights")
    B = tf.Variable(tf.random_uniform([k,10]), name="B-weights")
    W = tf.matmul(A, B, name="W-weights")

    A_hist = tf.summary.histogram("A-hist", A)
    B_hist = tf.summary.histogram("B-hist", B)
    W_hist = tf.summary.histogram("W-hist", B)

    b = tf.Variable(tf.zeros([10]), name="bias")
    b_hist = tf.summary.histogram("biases", b)
    with tf.name_scope("xA") as scope:
      Ax = tf.nn.softmax(tf.matmul(x,A))
    with tf.name_scope("xAB_b") as scope:
      y = tf.nn.softmax(tf.matmul(Ax,B) + b)
    
    y_hist = tf.summary.histogram("y", y)
    
    # Define loss and optimizer
    with tf.name_scope("regulization") as scope:
        regulize_A = tf.nn.l2_loss(A)
        regulize_B = tf.nn.l2_loss(B)
        regulize_W = tf.nn.l2_loss(W)

        rgA_summ = tf.summary.scalar("regulize A", regulize_A)
        rgB_summ = tf.summary.scalar("regulize B", regulize_B)
        rgW_summ = tf.summary.scalar("regulize W", regulize_W)

    y_ = tf.placeholder("float", [None,10], name="y-input")

    with tf.name_scope("xent") as scope:
      cross_entropy = -tf.reduce_sum(y_*tf.log(y))
      ce_summ = tf.summary.scalar("cross entropy", cross_entropy)
      loss = cross_entropy+(weight_decay*regulize_W)
      
    with tf.name_scope("train") as scope:
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate_starter, global_step,
                                learning_rate_decay_steps,learning_rate_decay_rate, staircase=True)
        # Passing global_step to minimize() will increment it at each step.
        
        #train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
        learning_step = ( tf.train.GradientDescentOptimizer(learning_rate)
                        .minimize(cross_entropy, global_step=global_step) )
        
        lr_summ = tf.summary.scalar("learning_rate", learning_rate)
    
    with tf.name_scope("test") as scope:
      correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
      accuracy_summary = tf.summary.scalar("accuracy", accuracy)
    
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("/tmp/mnist_logs", sess.graph_def)
    tf.initialize_all_variables().run()
    
    # Test trained model
    for i in range(10000000):
      if i % 10 == 0:  # Record summary data, and the accuracy
        feed = {x: mnist.test.images, y_: mnist.test.labels}
        result = sess.run([merged, accuracy], feed_dict=feed)
        summary_str = result[0]
        acc = result[1]
        writer.add_summary(summary_str, i)
        print("Accuracy at step %s: %s" % (i, acc))
      else:
        batch_xs, batch_ys = mnist.train.next_batch(100)
        feed = {x: batch_xs, y_: batch_ys}
        sess.run(learning_step, feed_dict=feed)
    
    print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))