# !/usr/bin/env python
# -*- coding:utf-8 -*-
# binnary classfy by logistic regression

__author__ = 'FesianXu'
__date__ = '2017/5/18'
__version__ = ''


import cv2
import tensorflow as tf
import numpy as np
import random
import time
import os

rng = np.random

pos_train_path = '../res/binary/train/8/'
neg_train_path = '../res/binary/train/9/'
pos_test_path = '../res/binary/test/8/'
neg_test_path = '../res/binary/test/9/'
learn_rate = 0.05
train_epochs = 25
batch_size = 60
train_time = 50
test_size = 30

def randomGetImages(root_path, size):
    dir = os.listdir(root_path)
    mat = np.array(np.zeros([size,32*64]))
    random_get = random.sample(dir,size)
    counter = 0
    for i in random_get:
        child = os.path.join('%s%s%s'% (root_path, '/', i))
        img = cv2.imread(child)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        retval, bin_img = cv2.threshold(gray, 120, 1, cv2.THRESH_BINARY)
        img = bin_img.reshape([1,32*64])
        mat[counter,:] = img[0]
        counter += 1
    return mat

if __name__ == '__main__':
    print('begin to test tensorflow!')
    timebegin = cv2.getTickCount()

    ### tf graph input
    x = tf.placeholder(tf.float32, [None, 32*64]) # none rows but 32*64 cols
    y = tf.placeholder(tf.float32, [None, 2])
    ### set model weight
    W = tf.Variable(tf.zeros([32*64, 2]))
    b = tf.Variable(tf.zeros([2]))

    ## construct model
    pred = tf.nn.softmax(tf.matmul(x, W) + b)
    # 交叉熵会出问题？交叉熵由于存在log函数，因此里面的变量需要加上一个很小的数比如0.0001，避免出现log0的情况
    # cost = tf.reduce_mean(tf.reduce_sum(tf.pow(pred-y,2), reduction_indices=1))
    delta = y*tf.log(pred+0.0001)
    cost = tf.reduce_mean(-tf.reduce_sum(delta, reduction_indices=1))
    optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:

        sess.run(init)
        print('W = ',sess.run(W),' b = ',sess.run(b))
        for epoch in range(train_epochs):
            avg_cost = 0
            for i in range(train_time):
                posimgs = randomGetImages(pos_train_path,batch_size)
                negimgs = randomGetImages(neg_train_path,batch_size)
                batch_xs = np.array(np.zeros([2*batch_size,32*64]))
                batch_xs[0:batch_size,:] = posimgs
                batch_xs[batch_size:,:] = negimgs
                batch_ys = np.array(np.zeros([2*batch_size,2]))
                batch_ys[0:batch_size,0] = 1
                batch_ys[batch_size:,1] = 1
                _, c = sess.run([optimizer,cost], feed_dict={x:batch_xs,
                                                             y:batch_ys})
                avg_cost += c/train_time

            print('time = ',epoch,'with cost = ',avg_cost)

        print('Finish')
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        test_pos = randomGetImages(pos_test_path,test_size)
        test_neg = randomGetImages(neg_test_path,test_size)
        test_imgs = np.array(np.zeros([2*test_size,32*64]))
        test_imgs[0:test_size,:] = test_pos
        test_imgs[test_size:,:] = test_neg
        labels = np.array(np.zeros([2*test_size,2]))
        labels[0:test_size,0] = 1
        labels[test_size:,1] = 1
        print('W = ',sess.run(W),' b = ',sess.run(b))
        print("Accuracy:", accuracy.eval({x: test_imgs, y: labels}))

    timeend = cv2.getTickCount()
    time = (timeend - timebegin)/ cv2.getTickFrequency()
    print('cost time = ',time,'s')