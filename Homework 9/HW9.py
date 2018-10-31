# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 23:48:28 2017

@author: 45336
"""
import numpy as np

import tensorflow as tf
from sklearn.datasets import load_breast_cancer


data = load_breast_cancer() # Loads the Wisconsin Breast Cancer dataset (569 examples in 30 dimensions)

# Parameters for the data
dim_data = 30
num_labels = 2
num_examples = 569

# Parameters for training
learning_rate = 1e-6
num_train = 400

#Define backtracking parameter alpha and beta:
alpha = 0.1
beta = 0.5


X = data['data'] # Data in rows
targets = data.target # 0-1 labels
labels = np.zeros((num_examples, num_labels))
for i in range(num_examples):
    labels[i,targets[i]]=1 # Conversion to one-hot representations


x = tf.placeholder(tf.float32, shape=[None, dim_data])
y_ = tf.placeholder(tf.float32, shape=[None, num_labels])

W = tf.Variable(tf.zeros([dim_data, num_labels]))

#b = tf.Variable(tf.zeros([num_labels]))
b = tf.Variable(tf.zeros(num_labels))

W_bt = tf.Variable(tf.zeros([dim_data, num_labels]))
#b_bt = tf.Variable(tf.zeros([num_labels]))
b_bt = tf.Variable(tf.zeros(num_labels))


y_prime = tf.matmul(x, W) + b
y = tf.nn.softmax(y_prime)

y_prime_bt = tf.matmul(x, W_bt) + b_bt


#accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

f = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_prime))
f_bt = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_prime_bt))



sess = tf.Session()
sess.run(tf.global_variables_initializer())

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_accuracy = sess.run(accuracy, feed_dict={x: X[:num_train, :], y_: labels[:num_train, :]})
train_cross_entropy = sess.run(f, feed_dict={x: X[:num_train, :], y_: labels[:num_train, :]})
print("Initial training accuracy %g, cross entropy %g" % (train_accuracy, train_cross_entropy))

for i in range(200):
    

    f0 = sess.run(f,feed_dict={x: X[:num_train, :], y_: labels[:num_train, :]})
    f0= float(f0)
    
    dWf = sess.run(tf.gradients(f, W), feed_dict={x: X[:num_train,:], y_: labe












