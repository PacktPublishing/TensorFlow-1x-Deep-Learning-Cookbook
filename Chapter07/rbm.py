import tensorflow as tf
import numpy as np
class RBM(object):
    def __init__(self, m, n):
        """
        m: Number of neurons in visible layer
        n: number of neurons in hidden layer
        """
        self._m = m
        self._n = n

        # Create the Computational graph

        # Weights and biases
        self._W = tf.Variable(tf.random_normal(shape=(self._m, self._n)))
        self._c = tf.Variable(np.zeros(self._n).astype(np.float32))  # bias for hidden layer
        self._b = tf.Variable(np.zeros(self._m).astype(np.float32))  # bias for Visible layer

        # Placeholder for inputs
        self._X = tf.placeholder('float', [None, self._m])

        # Forwards Pass
        _h = tf.nn.sigmoid(tf.matmul(self._X, self._W) + self._c)
        self.h = tf.nn.relu(tf.sign(_h - tf.random_uniform(tf.shape(_h))))

        # Backward pass
        _v = tf.nn.sigmoid(tf.matmul(self.h, tf.transpose(self._W)) + self._b)
        self.V = tf.nn.relu(tf.sign(_v - tf.random_uniform(tf.shape(_v))))


        # Objective Function
        objective = tf.reduce_mean(self.free_energy(self._X)) - tf.reduce_mean(
            self.free_energy(self.V))
        self._train_op = tf.train.GradientDescentOptimizer(1e-3).minimize(objective)

        # Cross entropy cost
        reconstructed_input = self.one_pass(self._X)
        self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self._X, logits=reconstructed_input))

    def set_session(self, session):
        self.session = session

    def free_energy(self, V):
        b = tf.reshape(self._b, (self._m, 1))
        term_1 = -tf.matmul(V, b)
        term_1 = tf.reshape(term_1, (-1,))

        term_2 = -tf.reduce_sum(tf.nn.softplus(tf.matmul(V, self._W) + self._c))
        return term_1 + term_2

    def one_pass(self, X):
        W = (self._W)
        c = (self._c)
        h = tf.nn.sigmoid(tf.matmul(X, W) + c)
        return (tf.matmul(h, tf.transpose(self._W)) + self._b)

    def reconstruct(self, X):
        x = tf.nn.sigmoid(self.one_pass(X))
        return self.session.run(x, feed_dict={self._X: X})

    def fit(self, X, epochs=1, batch_size=100):
        N, D = X.shape
        num_batches = N // batch_size
        print('epochs')
        obj = []
        for i in range(epochs):
            # X = shuffle(X)
            for j in range(num_batches):
                batch = X[j * batch_size: (j * batch_size + batch_size)]
                _, ob = self.session.run([self._train_op, self.cost], feed_dict={self._X: batch})
            #if i % 10 == 0:
            print('training epoch {0} cost {1}'.format(i, ob))
            obj.append(ob)
        return obj