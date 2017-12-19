import tensorflow as tf


# v_1 = tf.constant([1,2,3,4], name='v_1')
# v_2 = tf.constant([2,1,5,3], name='v_2')
# v_add = tf.add(v_1,v_2)

# with tf.Session() as sess:
#     writer = tf.summary.FileWriter('graphs', sess.graph)
#     print(sess.run([v_1, v_2,v_add]))
#
# writer.close()

sess = tf.InteractiveSession()

v_1 = tf.constant([1,2,3,4])
v_2 = tf.constant([2,1,5,3])
I_matrix = tf.eye(5)
v_add = v_1 + v_2 #tf.add(v_1,v_2)

print(v_add.eval())
print(I_matrix.eval())

sess.close()
