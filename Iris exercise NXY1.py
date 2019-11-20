# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 21:19:25 2019

@author: LENOVO
"""

import sklearn.datasets as datasets
import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt

Iris = datasets.load_iris()
x = Iris.data
y = Iris.target 
#yi = Iris.target [:, np.newaxis]
x_data = x/x.max()
y_data_iris = np.array([[int(k) for k in np.arange(3) == i] for i in y])

    
def add_layer(input,input_size,out_size,activate_function=None):
    Weights = tf.Variable(tf.truncated_normal(shape=[input_size,out_size]))
    bias = tf.Variable(tf.zeros(shape=[out_size])+0.1)
    output = tf.matmul(input, Weights)+bias
    if activate_function==None:
        return output
    else:
        return activate_function(output)

xs = tf.placeholder(shape=[None,4], dtype = tf.float32)
ys = tf.placeholder(shape=[None,3], dtype = tf.float32)
layer1 = add_layer(xs,4,30,activate_function=tf.nn.relu)
layer2 = add_layer(layer1, 30, 10, activate_function=tf.nn.relu)
predict = add_layer(layer2,10,3)
prediction = tf.nn.softmax(predict)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))

optimizer = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(100000):
        sess.run(optimizer,feed_dict={xs:x_data,ys:y_data})
        if i % 50 == 0:
            myloss = sess.run(cross_entropy,feed_dict={xs:x_data,ys:y_data})
            y_predict = sess.run(prediction,feed_dict={xs:x_data})
            prediction_num = np.array([np.argmax(i) for i in y_predict])
            true_num = np.sum([int(prediction_num[i] == y_data_label[i]) for i in range(y_data_label.shape[0])])
            accuracy = true_num / y_data_label.shape[0]
            print(y_predict)
            print(myloss) 
            print(accuracy)