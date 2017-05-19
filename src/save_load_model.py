# !/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'FesianXu'
__date__ = '2017/5/19'
__version__ = ''

"""
using to show how to save the trained model and load it
This demo is based on and modified from multi_classfy.py
"""

import cv2
import tensorflow as tf
import numpy as np
import random
import time
import os

c1_train_path = '../res/multi/train/c1/'
c2_train_path = '../res/multi/train/c2/'
c3_train_path = '../res/multi/train/c3/'
c1_test_path = '../res/multi/test/c1/'
c2_test_path = '../res/multi/test/c2/'
c3_test_path = '../res/multi/test/c3/'

learning_rate = 0.25
train_batch = 50
training_time = 5
train_epoch = 30
test_batch = 30

def randomGetImages(root_path, size):
    '''
    :param root_path: 文件根目录
    :param size:  随机选择的图片数量
    :return: 文件夹内的图片矩阵，行优先
    '''
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

def getSamplesMat():
    '''
    :return: mats one-hot 数据矩阵
    :return: labels one-hot 标签
    '''
    c1_imgs = randomGetImages(c1_train_path, train_batch)
    c2_imgs = randomGetImages(c2_train_path, train_batch)
    c3_imgs = randomGetImages(c3_train_path, train_batch)
    mats = np.array(np.zeros([3*train_batch,32*64]))
    labels = np.array(np.zeros([3*train_batch,3]))
    mats[0:train_batch, :] = c1_imgs
    mats[train_batch:2*train_batch,:] = c2_imgs
    mats[2*train_batch:3*train_batch, :] = c3_imgs
    labels[0:train_batch,0] = 1
    labels[train_batch:2*train_batch, 1] = 1
    labels[2*train_batch:3*train_batch,2] = 1
    return mats,labels

def getTestBatch():
    c1_test = randomGetImages(c1_test_path, test_batch)
    c2_test = randomGetImages(c2_test_path, test_batch)
    c3_test = randomGetImages(c3_test_path, test_batch)
    mats = np.array(np.zeros([3*test_batch, 32*64]))
    mats[0:test_batch,:] = c1_test
    mats[test_batch:2*test_batch,:] = c2_test
    mats[2*test_batch:3*test_batch, :] = c3_test
    labels = np.array(np.zeros([3*test_batch, 3]))
    labels[0:test_batch, 0] = 1
    labels[test_batch:2*test_batch, 1] = 1
    labels[2*test_batch: 3*test_batch, 2] = 1
    return mats,labels


if __name__ == '__main__':
    print('begin to multiple-class trainning!')
    timebegin = cv2.getTickCount()
    # graph input
    x = tf.placeholder(tf.float32, [None, 32*64])
    y = tf.placeholder(tf.float32, [None, 3])

    # model parameters
    W = tf.Variable(tf.zeros([32*64, 3]))
    b = tf.Variable(tf.zeros([3]))

    pred = tf.nn.softmax(tf.matmul(x, W) + b)
    cost_func = y*tf.log(pred+0.0001)
    cost = tf.reduce_mean(-tf.reduce_sum(cost_func, reduction_indices=1))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver() ## saver
    with tf.Session() as sess:
        saver.restore(sess, './model/model.ckpt')
        ### load the model which has been trainned before
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        test_mats,test_labels = getTestBatch()
        print("Accuracy:", accuracy.eval({x: test_mats, y: test_labels}))
        # saver_path = saver.save(sess, './model/model.ckpt')
        # print(saver_path)
        ## save model

    timeend = cv2.getTickCount()
    time = (timeend - timebegin)/ cv2.getTickFrequency()
    print('cost time = ',time,'s')

