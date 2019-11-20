# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 16:26:19 2019

@author: LENOVO
"""

import sklearn.datasets as datasets
import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt

Boston = datasets.load_boston()  
x = Boston.data
y = Boston.target [:, np.newaxis]
x_data = x/x.max()
y_data = y/y.max()

def add_layer (input, input_size, out_size, activate_function=None):
    weights = tf.Variable (tf.truncated_normal(shape = [input_size,out_size]))
    bias = tf.Variable(tf.zeros(shape=[out_size])+0.1)
    output= tf.matmul(input, weights)+bias
    if activate_function==None:
        return output
    else:
        return activate_function(output)
    
xs = tf.placeholder(shape=[None,13], dtype = tf.float32)
ys = tf.placeholder(shape=[None,1], dtype = tf.float32)
layer1 = add_layer(xs, 13,30,tf.nn.relu)
layer2 = add_layer(layer1, 30, 10, tf.nn.relu)
predict = add_layer(layer2, 10, 1)


loss = tf.reduce_mean((ys-predict)**2)

optimizer = tf.train.GradientDescentOptimizer(0.005).minimize(loss)

init = tf.global_variables_initializer()
plt.ion()

with tf.Session() as sess:
    sess.run(init)
    for i in range (100000):
        sess.run(optimizer, feed_dict={xs:x_data, ys:y_data})
        if i % 50==0:
            myloss = sess.run(loss, feed_dict={xs:x_data, ys:y_data})
            print (myloss)
            y_predict = sess.run(predict, feed_dict = {xs:x_data})
            plt.cla()
            plt.scatter(range(y_data.shape[0]),y_data)
            plt.plot(range(y_data.shape[0] ),y_predict, 'r-', lw=5)
            plt.pause(0.5)
    
