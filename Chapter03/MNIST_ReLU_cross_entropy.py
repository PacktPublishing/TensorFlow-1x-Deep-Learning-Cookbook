import tensorflow as tf
import tensorflow.contrib.layers as layers

from tensorflow.python import debug as tf_debug

# Network Parameters
n_hidden = 30
n_classes = 10
n_input = 784

# Hyperparameters
batch_size = 200
eta = 0.001
max_epoch = 10

# MNIST input data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

def multilayer_perceptron(x):
    fc1 = layers.fully_connected(x, n_hidden, activation_fn=tf.nn.relu, scope='fc1')
    #fc2 = layers.fully_connected(fc1, 256, activation_fn=tf.nn.relu, scope='fc2')
    out = layers.fully_connected(fc1, n_classes, activation_fn=None, scope='out')
    return out

# build model, loss, and train op
x = tf.placeholder(tf.float32, [None, n_input], name='placeholder_x')
y = tf.placeholder(tf.float32, [None, n_classes], name='placeholder_y')
y_hat = multilayer_perceptron(x)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=y))
train = tf.train.AdamOptimizer(learning_rate= eta).minimize(loss)
init = tf.global_variables_initializer()




with tf.Session() as sess:
    sess.run(init)
    for epoch in range(10):
        epoch_loss = 0.0
        batch_steps = int(mnist.train.num_examples / batch_size)
        for i in range(batch_steps):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run([train, loss],
                               feed_dict={x: batch_x, y: batch_y})
            epoch_loss += c / batch_steps
        print ('Epoch %02d, Loss = %.6f' % (epoch, epoch_loss))

    # Test model
    correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print ('Accuracy%:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels})*100)
