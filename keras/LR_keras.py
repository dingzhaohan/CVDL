import numpy as np
import pandas as pd
import tensorflow as tf

W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)

loss = tf.reduce_mean(tf.square(linear_model - y))
optimizer = tf.train.GradientDescentOptimizer(0.0001)
train = optimizer.minimize(loss)

df = pd.read_csv("data.csv", sep="\t", header=None)
points = df.values
x_train = [float(p[0]) for p in points]
y_train = [float(p[1]) for p in points]

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(100000):
    sess.run(train,{x:x_train, y:y_train})
    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train})
    print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))
