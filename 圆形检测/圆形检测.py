#!/usr/bin/python
#coding=utf-8
''' face detect convolution'''
# pylint: disable=invalid-name
import os
import logging as log
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import cv2
import dealw
SIZE = 64

def weightVariable(shape):
    ''' build weight variable'''
    init = tf.random_normal(shape, stddev=0.01)
    #init = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(init)

def biasVariable(shape):
    ''' build bias variable'''
    init = tf.random_normal(shape)
    return tf.Variable(init)

def conv2d(x, W):
    ''' conv2d by 1, 1, 1, 1'''
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxPool(x):
    ''' max pooling'''
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def dropout(x, keep):
    ''' drop out'''
    return tf.nn.dropout(x, keep)

x_data = tf.placeholder(tf.float32, [None, SIZE, SIZE, 3])
y_data = tf.placeholder(tf.float32, [None, 2])

keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)
   

W1 = weightVariable([5, 5, 3, 32]) # 卷积核大小(3,3)， 输入通道(3)， 输出通道(32)
b1 = biasVariable([32])
conv1 = tf.nn.relu(conv2d(x_data, W1) + b1)
pool1 = maxPool(conv1)
drop1 = dropout(pool1, keep_prob_5)

W2 = weightVariable([5, 5, 32, 64])
b2 = biasVariable([64])
conv2 = tf.nn.relu(conv2d(drop1, W2) + b2)
pool2 = maxPool(conv2)
drop2 = dropout(pool2, keep_prob_5)

W3 = weightVariable([5, 5, 64, 64])
b3 = biasVariable([64])
conv3 = tf.nn.relu(conv2d(drop2, W3) + b3)
pool3 = maxPool(conv3)
drop3 = dropout(pool3, keep_prob_5)

temp = tf.reshape(drop3, [-1, 8 * 8 * 64])
wf1 = weightVariable([8 * 8 * 64, 512])
bf1 = biasVariable([512])
hf1 = tf.nn.relu(tf.matmul(temp, wf1) + bf1)
dropf = dropout(hf1, keep_prob_75)


wf2 = weightVariable([512, 2])
bf2 = biasVariable([2])
hf2 = tf.add(tf.matmul(dropf, wf2), bf2)
out = hf2

train_x, train_y = dealw.getcircle()
train_x = np.array(train_x)
train_x = train_x / 255
train_y = np.array(train_y)

#误差震荡是正常情况，最终一般都能收敛

#交叉熵
temp = tf.nn.softmax_cross_entropy_with_logits_v2(logits=out, labels=y_data)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=out, labels=y_data))

#交叉熵, 分开写           当tf.log(out)时， out值可能很小,即使加上一个很小的常数也不行，会导致误差一直不变， 此时需要改变学习速率，从0.01 到 0.001
#out = tf.nn.softmax(out)
#temp = -tf.reduce_sum(y_data*tf.log(out),axis = 1)
#cross_entropy = tf.reduce_mean(temp)

#交叉熵
#out = tf.nn.softmax(out)
#temp = -y_data*tf.log(out)
#cross_entropy = tf.reduce_mean(temp)

train_step = tf.train.AdamOptimizer(0.1).minimize(cross_entropy)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y_data, 1)), tf.float32))

saver=tf.train.Saver(max_to_keep=1)  #保存模型

with tf.Session() as sess:

        #sess.run(tf.global_variables_initializer())
        model_file=tf.train.latest_checkpoint('./')
        saver.restore(sess,model_file)
        for n in range(100):

            for i in range(1):
                p = np.random.randint(100)-40
                _, loss,Temp= sess.run([train_step, cross_entropy,temp], feed_dict = {x_data: train_x[p:p+30], y_data: train_y[p:p+30], keep_prob_5:0.75, keep_prob_75:0.75})
            if n % 20 == 0:
                print(loss)
                print("\n")
                acc = accuracy.eval({x_data:train_x, y_data:train_y, keep_prob_5:1.0, keep_prob_75:1.0})
                print(acc)
        saver.save(sess,'./test.ckpt',global_step=i+1)
   
