# scaling was really important
# hidden_layer == output_dim

import time
import math
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import tensorflow as tf
import utils

df = utils.data_read()
seq_length = 20
data_dim = 6
output_dim = 1
X_train, y_train, X_test, y_test = utils.preprocessing_data(df[:: -1], seq_length)


print("X_train", X_train.shape) # num, 20, 6
print("y_train", y_train.shape) # num 1
print("X_test", X_test.shape) # num 20 6
print("y_test", y_test.shape) # num 1


X = tf.placeholder(tf.float32, [None, seq_length, data_dim]) # num, 20, 6
Y = tf.placeholder(tf.float32, [None,1]) # num, 1

#rnn architecture here
rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [128, 128]]

cells = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
outputs, _states = tf.nn.dynamic_rnn(cells, X, dtype = tf.float32)

Y_pred = tf.contrib.layers.fully_connected(outputs[:,-1], output_dim, activation_fn = None)
loss = tf.reduce_sum(tf.square(Y_pred - Y))
train = tf.train.AdamOptimizer(0.01).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(1000):
    _, l = sess.run([train, loss],
                   feed_dict = {X:X_train, Y:y_train})
    print(i, l)

#plotting
testPredict = sess.run(Y_pred, feed_dict={X:X_test})
plt.plot(y_test)
plt.plot(testPredict)
plt.show()