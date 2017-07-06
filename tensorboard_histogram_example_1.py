# -*- coding: utf-8 -*-
import tensorflow as tf

k = tf.placeholder(tf.float32)

# Make a normal distribution, with a shifting mean
mean_moving_normal = tf.random_normal(shape=[1000], mean=(15*k), stddev=1)
# Record that distribution into a histogram summary
summaries=tf.summary.histogram("normal/moving_mean", mean_moving_normal)#官网此处漏了summaries变量


# Setup a session and summary writer
sess = tf.Session()
writer = tf.summary.FileWriter("/tmp/histogram_example1")

# Setup a loop and write the summaries to disk
N = 400
for step in range(N):
  k_val = step/float(N)
  summ = sess.run(summaries, feed_dict={k:k_val})
  writer.add_summary(summ, global_step=step)
