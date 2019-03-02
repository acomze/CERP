import socket
import threading
import time
import sys
import os
import struct
import pickle as P
import argparse
import sys
import alexnet
import cv2
import tensorflow as tf
import numpy as np
import caffe_classes
import time

BUFFER_SIZE = 1024



def socket_service():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('0.0.0.0', 50000))
        s.listen(10)
    except socket.error as msg:
        print (msg)
        sys.exit(1)
    print ('Waiting connection...')

    while 1:
        conn, addr = s.accept()
        t = threading.Thread(target=deal_data, args=(conn, addr))
        t.start()

def deal_data(conn, addr):
    print ('Accept new connection from {0}'.format(addr))
    #conn.settimeout(500)
    conn.send(b"Hi, Welcome to the server!")

    while 1:
        # fileinfo_size = struct.calcsize('128sl')
        # buf = conn.recv(1024000000)
        buf = b""
        while True:
            packet = conn.recv(1024)
            if not packet: break
            buf += packet
 
        print(len(buf))
        testImg = {}
        if buf:
            bytes_list = P.loads(buf)
            start_time = time.time()
            pic = 1
            for img_buf in bytes_list:
                print(len(img_buf))
                img_buf = np.asarray(bytearray(img_buf), dtype=np.uint8)
                # img_data_ndarray = cv2.imdecode(file_bytes, cv2.CV_LOAD_IMAGE_UNCHANGED)
                img = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)
                # print(img.shape)
                testImg[pic] = img
                pic += 1
            
            dropoutPro = 1
            classNum = 1000
            skip = []

            imgMean = np.array([104, 117, 124], np.float)
            x = tf.placeholder("float", [1, 227, 227, 3])

            model = alexnet.alexNet(x, dropoutPro, classNum, skip)
            score = model.fc3
            softmax = tf.nn.softmax(score)
            result = {}
            # print(testImg)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                model.loadModel(sess)

                for key, img in testImg.items():
                    # img preprocess
                    resized = cv2.resize(img.astype(np.float), (227, 227)) - imgMean
                    maxx = np.argmax(sess.run(softmax, feed_dict={x: resized.reshape((1, 227, 227, 3))}))
                    res = caffe_classes.class_names[maxx]

                    # font = cv2.FONT_HERSHEY_SIMPLEX
                    # cv2.putText(img, res, (int(img.shape[0]/3), int(img.shape[1]/3)), font, 1, (0, 255, 0), 2)
                    # print("{}: {}\n----".format(key, res))
                    print(key, res)
                    result[key] = res
            end_time = time.time()
            print("computation_time", end_time - start_time)
            # conn.send(P.dumps(result))

            fr = P.dumps(result)
            file_size = len(fr)
            sent_size = 0
            while sent_size < file_size:
                remained_size = file_size - sent_size
                send_size = BUFFER_SIZE if remained_size > BUFFER_SIZE else remained_size
                send_file = fr.read(send_size)
                sent_size += send_size
                conn.send(send_file)
                rback = conn.recv(2)


        conn.close()
        break


if __name__ == '__main__':
    socket_service()
