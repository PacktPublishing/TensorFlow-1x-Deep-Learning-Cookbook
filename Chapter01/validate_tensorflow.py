import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  ## To deactivate SSE Warnings
message = tf.constant('Welcome to the exciting world of Deep Neural Networks')
#print(message)

with tf.Session() as sess:
    writer = tf.summary.FileWriter('\graphs',sess.graph) # To activate Tensorboard
    print(sess.run(message).decode())

writer.close()