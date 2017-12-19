import tensorflow as tf

# Create two random matrices
a = tf.Variable(tf.random_normal([4,5], stddev=2))
b = tf.Variable(tf.random_normal([4,5], stddev=2))

#Element Wise Multiplication
A = a * b

#Multiplication with a scalar 2
B = tf.scalar_mul(2, A)

# Elementwise division, its result is
C = tf.div(a,b)

#Element Wise remainder of division
D = tf.mod(a,b)

# Pairwise cross product
#E = tf.cross(a,b)
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    writer = tf.summary.FileWriter('graphs', sess.graph)
    a,b,A_R, B_R, C_R, D_R = sess.run([a , b, A, B, C, D])
    print("a\n",a,"\nb\n",b, "a*b\n", A_R, "\n2*a*b\n", B_R, "\na/b\n", C_R, "\na%b\n", D_R)

writer.close()

