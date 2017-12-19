import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  ## To deactivate SSE Warnings
# Create a graph

# Selecting only CPU
with tf.device('/cpu:0'):
    rand_t = tf.random_uniform([50, 50], 0, 10, dtype=tf.float32, seed=0)
    a = tf.Variable(rand_t)
    b = tf.Variable(rand_t)
    c = tf.matmul(a, b)
    init = tf.global_variables_initializer()
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
sess.run(init)
print(sess.run(c))
sess.close()

# Selecting only GPU
with tf.device('/gpu:0'):
    rand_t = tf.random_uniform([50, 50], 0, 10, dtype=tf.float32, seed=0)
    a = tf.Variable(rand_t)
    b = tf.Variable(rand_t)
    c = tf.matmul(a, b)
    init = tf.global_variables_initializer()
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
sess.run(init)
print(sess.run(c))
sess.close()

# selecting Multiple GPUs
c=[]
for d in ['/gpu:1','/gpu:2']:
    with tf.device(d):
        rand_t = tf.random_uniform([50, 50], 0, 10, dtype=tf.float32, seed=0)
        a = tf.Variable(rand_t)
        b = tf.Variable(rand_t)
        c.append(tf.matmul(a,b))
        init = tf.global_variables_initializer()

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
sess.run(init)
print(sess.run(c))
sess.close()