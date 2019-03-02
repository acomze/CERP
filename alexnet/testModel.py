#!/usr/bin/env python
# coding: UTF-8

import os
import argparse
import sys
import alexnet
import cv2
import tensorflow as tf
import numpy as np
import caffe_classes

parser = argparse.ArgumentParser(description='Classify some images.')
parser.add_argument('-m', '--mode', choices=['folder', 'url'], default='folder')
parser.add_argument('-p', '--path', help='Specify a path [e.g. testModel]', default = 'testModel')
args = parser.parse_args(sys.argv[1:])

if args.mode == 'folder':
    #get testImage
    withPath = lambda f: '{}/{}'.format(args.path,f)
    testImg = dict((f,cv2.imread(withPath(f))) for f in os.listdir(args.path) if os.path.isfile(withPath(f)))


# noinspection PyUnboundLocalVariable
if testImg.values():
    #some params
    dropoutPro = 1
    classNum = 1000
    skip = []

    imgMean = np.array([104, 117, 124], np.float)
    x = tf.placeholder("float", [1, 227, 227, 3])

    model = alexnet.alexNet(x, dropoutPro, classNum, skip)
    score = model.fc3
    softmax = tf.nn.softmax(score)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.loadModel(sess)

        for key,img in testImg.items():
            #img preprocess
            resized = cv2.resize(img.astype(np.float), (227, 227)) - imgMean
            maxx = np.argmax(sess.run(softmax, feed_dict = {x: resized.reshape((1, 227, 227, 3))}))
            res = caffe_classes.class_names[maxx]

            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(img, res, (int(img.shape[0]/3), int(img.shape[1]/3)), font, 1, (0, 255, 0), 2)
            print("{}: {}\n----".format(key,res))
            # cv2.imshow("demo", img)
            # cv2.waitKey(0)
