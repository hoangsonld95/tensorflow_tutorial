import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

w = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = w*x + b

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(linear_model, {x:[1,2,3,4]}))

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

fix_w = tf.assign(w, [-1.])
fix_b = tf.assign(b, [1.])

