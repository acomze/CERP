import socket
import threading
import time
import hashlib
import struct
# import dill
import os
import cv2
import caffe_classes
import numpy as np
import tensorflow as tf
# import prettytable as pt
import alexnet


BUFFER_SIZE = 1024
HEAD_STRUCT = '128sIq32s'
info_size = struct.calcsize(HEAD_STRUCT)

def cal_md5(file_path):
    with open(file_path, 'rb') as fr:
        md5 = hashlib.md5()
        md5.update(fr.read())
        md5 = md5.hexdigest()
        return md5

def unpack_file_info(file_info):
    file_name, file_name_len, file_size, md5 = struct.unpack(HEAD_STRUCT, file_info)
    file_name = file_name[:file_name_len]
    return file_name, file_size, md5

class server(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.load_model()

    def load_model(self):
        # Initialize parameters for AlexNet
        dropoutPro = 1
        classNum = 1000
        skip = []
        self.imgMean = np.array([104, 117, 124], np.float)
        self.x = tf.placeholder("float", [1, 227, 227, 3])
        model = alexnet.alexNet(self.x, dropoutPro, classNum, skip)
        score = model.fc3
        self.softmax = tf.nn.softmax(score)
        
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        config=tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        model.loadModel(self.sess)


    def run(self, ip = "127.0.0.1", port = 500000):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.connect((ip, port))            
            print("Server: Connection Succeed")
        except:
            print("Server: Connection ERROR")
            exit()
        
        # Check the connection
        isready = s.recv(BUFFER_SIZE)
        print(isready)
        s.send(b"Server: READY to receive the File Information")

        # Check the file to transfer
        datasize = 0
        file_info = s.recv(info_size)
        file_name, file_size, md5_recv = unpack_file_info(file_info)
        datasize = file_size
        s.send(b"Server: READY to receive the File")

        # Receive data from client 
        received_size = 0
        with open("receive/002689.jpg", 'wb') as fw:
            while received_size < file_size:
                # print received_size
                remained_size = file_size - received_size
                recv_size = BUFFER_SIZE if remained_size > BUFFER_SIZE else remained_size
                recv_file = s.recv(recv_size)
                s.send(b"Server: OK")
                received_size += recv_size
                fw.write(recv_file)
            fw.close()
        md5 = cal_md5("receive/002689.jpg")
        if md5 != md5_recv:
            print("Server: MD5 compared fail!")
        else:
            print("Server: Received successfully")

        # Run AlexNet and collect the result
        imagePath = "./receive"
        imageName = [
            "002689.jpg"
        ]
        withPath = lambda imgName: '{}/{}'.format(imagePath,imgName)
        testImg = dict((imgName,cv2.imread(withPath(imgName))) for imgName in imageName)
                
        for key,img in testImg.items(): 
            resized = cv2.resize(img.astype(np.float), (227, 227)) - self.imgMean
            maxx = np.argmax(self.sess.run(self.softmax, feed_dict = {self.x: resized.reshape((1, 227, 227, 3))}))
            result = caffe_classes.class_names[maxx]
            print("{}: {}\n----".format(key,result))
            s.send(bytes(result.encode('utf-8')))
        s.close()


if __name__ == '__main__':
    print("Server: Test start...")
    s = server()
    ip = "192.168.1.199"
    # ip = "127.0.0.1"
    port = 50000
    s.run(ip, port)