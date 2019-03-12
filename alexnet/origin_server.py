import socket
import threading
import time
import hashlib
import struct
# import dill
import os
# import cv2
import caffe_classes
import numpy as np
import tensorflow as tf
# import prettytable as pt
import alexnet
from PIL import Image


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
    return file_name.decode(), file_size, md5

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


    def run(self, ip = "127.0.0.1", port = 50000):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.connect((ip, port))            
            print("Server: Connection Succeed")
        except Exception as e:
            print("Server: Connection ERROR")
            print("Exception: ", repr(e))
            exit()
        
        # Check the connection
        isready = s.recv(BUFFER_SIZE)
        print(isready)
        s.send(b"Server: READY to receive the File Information")

        receive_path = "./receive"
        image_list = []
        file_num = s.recv(BUFFER_SIZE)
        file_count = int.from_bytes(file_num, byteorder='big', signed = False)
        print("File num: ",file_num, "/",file_count)
        while file_count > 0:
            file_count -= 1

            # Check the file to transfer 
            file_info = s.recv(info_size)
            file_name, file_size, md5_recv = unpack_file_info(file_info)
            image_list.append(file_name)
            s.send(b"Server: Received file information")

            # Receive data from client 
            received_size = 0
            with open(receive_path+'/'+file_name,'wb') as fw:
                while received_size < file_size:
                    # print received_size
                    remained_size = file_size - received_size
                    recv_size = BUFFER_SIZE if remained_size > BUFFER_SIZE else remained_size
                    recv_file = s.recv(recv_size)
                    s.send(b"Server: OK")
                    received_size += recv_size
                    fw.write(recv_file)
                fw.close()
            # md5 = cal_md5(receive_path+'/'+file_name)
            # if md5 != md5_recv:
            #     print("Server: {} MD5 compared fail!".format(file_name))
            #     s.send(b"No")
            # else:
            #     print("Server: Received {} successfully".format(file_name))
            #     s.send(b"OK")
            print("Server: Received {} successfully".format(file_name))
            s.send(b"OK")

        # Run AlexNet and collect the result
        # withPath = lambda imgName: '{}/{}'.format(receive_path,image_list)
        # testImg = dict((imgName,cv2.imread(receive_path+'/'+imgName)) for imgName in image_list)
        testImg = dict((imgName,Image.open(receive_path+'/'+imgName) ) for imgName in image_list)
        # print(testImg)

        for imgName,img in testImg.items(): 
            # resized = cv2.resize(img.astype(np.float), (227, 227)) - self.imgMean
            resized = np.array(img.resize((227, 227))) - self.imgMean
            maxx = np.argmax(self.sess.run(self.softmax, feed_dict = {self.x: resized.reshape((1, 227, 227, 3))}))
            # maxx = 0 # for test use
            result = caffe_classes.class_names[maxx]
            print("{}: {}\n----".format(imgName,result))
            s.send(bytes(imgName.encode('utf-8')))
            s.recv(2)
            s.send(bytes(result.encode('utf-8')))
        s.close()


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    print("Server: Test start...")
    s = server()
    # ip = "192.168.1.128" # Raspberry
    # ip = "192.168.1.101" # Jetson
    # ip = "192.168.1.199" # Desktop
    ip = "127.0.0.1" # Local
    port = 50000
    s.run(ip, port)