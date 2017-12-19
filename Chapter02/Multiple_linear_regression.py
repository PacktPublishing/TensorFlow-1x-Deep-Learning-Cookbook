import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X - mean) / std
    return X
# Data
boston = tf.contrib.learn.datasets.load_dataset('boston')
X_train, Y_train = boston.data, boston.target
X_train = normalize(X_train)
m = len(X_train)  #Number of training examples
n = 13   # Number of features

#X_train = normalize(X_train)

#print(X_train)

# Placeholder for the Training Data
X = tf.placeholder(tf.float32, name='X', shape=[m,n])
Y = tf.placeholder(tf.float32, name='Y')

# Variables for coefficients
b = tf.Variable(0.0)
w = tf.Variable(tf.random_normal([n,1]))


# The Linear Regression Model
Y_hat = tf.matmul(X, w) + b

# Loss function
loss = tf.reduce_mean(tf.square(Y - Y_hat, name='loss')) + 0.6*tf.nn.l2_loss(w)

# Gradient Descent with learning rate of 0.05 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss)

# Initializing Variables
init_op = tf.global_variables_initializer()
total = []
# Computation Graph
with tf.Session() as sess:
    # Initialize variables
    sess.run(init_op)
    writer = tf.summary.FileWriter('graphs', sess.graph)

    # train the model for 100 epcohs
    for i in range(100):
       _, l = sess.run([optimizer, loss], feed_dict={X: X_train, Y: Y_train})
       total.append(l)
       print('Epoch {0}: Loss {1}'.format(i, l))

    writer.close()

    w_value, b_value = sess.run([w, b])

#print(w_value, b_value)
N= 500
X_new = X_train [N,:]
Y_pred =  (np.matmul(X_new, w_value) + b_value).round(1)
print('Predicted value: ${0}  Actual value: ${1}'.format(Y_pred[0]*1000, Y_train[N]*1000) , '\nDone')
# Plot the result
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(X_train[0], X_train[1], Y_train)
# plt.plot(X_train, Y_train, 'bo', label='Real Data')
# plt.plot(X_train,Y_pred,  'r', label='Predicted Data')
# plt.legend()
# plt.show()
#
plt.plot(total)
plt.show()

