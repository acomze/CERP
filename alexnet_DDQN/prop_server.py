import socket
import threading
import time
import hashlib
import struct
import os
import caffe_classes
import numpy as np
import tensorflow as tf
import alexnet
from PIL import Image
import psutil

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
BUFFER_SIZE = 1200
HEAD_STRUCT = '128sIqi32xs'
info_size = struct.calcsize(HEAD_STRUCT)

###################
# FileProcessor: 
# Functions to deal with files
###################
class FileProcessor():
    def cal_md5(self, file_path):
        with open(file_path, 'rb') as fr:
            md5 = hashlib.md5()
            md5.update(fr.read())
            md5 = md5.hexdigest()
            return md5

    def unpack_file_info(self, file_info):
        file_name, file_name_len, file_size, required_index, md5 = struct.unpack(HEAD_STRUCT, file_info)
        file_name = file_name[:file_name_len]
        return file_name.decode(), file_size, required_index, md5

    def get_local_info(self):
        cpu_percent = psutil.cpu_percent(None)
        conn_type = 1
        local_info = struct.pack(HEAD_STRUCT, cpu_percent, conn_type)
        return local_info

###################
# Server: 
# The server. See notes on each function.
###################    
class Server(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.load_model()
        self.file_processor = FileProcessor()
        with open("sprintGo.txt","r") as ulFile:
            self.size_arrange = list(int(float(ul)/8) for ul in ulFile.readlines())
            ulFile.close()

    ## Configure the alexNet and load the pre-trained model
    def load_model(self):
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

    ## Run the server
    def run(self, sock, addr):
        print("Server: Connecting %s:%s..." % addr)
        print("Server: Here is the server side")

        sock.send(b"Server: Ready")
        # isReady = sock.recv(BUFFER_SIZE)
        # print(isReady)
        packet_receive = sock.recv(BUFFER_SIZE)
        
        # "Resource query": Proping processs
        # "File transfer": Transfering process
        print(packet_receive)
        if packet_receive.decode() == "Resource query":
            local_info = self.file_processor.get_local_info()
            sock.send(local_info)
        elif packet_receive.decode() == "File transfer":
            receive_path = "./receive"
            image_list = []
            file_num = sock.recv(BUFFER_SIZE)
            file_count = int.from_bytes(file_num, byteorder='big', signed = False)
            print("File num: ",file_num, "/",file_count)

            while file_count > 0:
                file_count -= 1

                # Check the file to transfer 
                file_info = sock.recv(info_size)
                file_name, file_size, self.required_index, md5_recv = self.file_processor.unpack_file_info(file_info)
                image_list.append(file_name)
                sock.send(b"S: Received file information")

                # Receive data from client 
                received_size = 0
                require_size = self.size_arrange[0]
                print("S: receiving image {}...".format(file_name))
                with open(receive_path+'/'+file_name,'wb') as fw:
                    count = 0
                    # required_index = 0
                    while received_size < file_size:
                        remained_size = file_size - received_size
                        require_size = self.size_arrange[self.required_index]
                        self.required_index += 1
                        recv_size = min(require_size, remained_size)
                        recv_file = sock.recv(recv_size)
                        # print(len(recv_file))
                        # print(recv_file) 
                        print("[{}kB/s] Server: receiving file packet {}...".format(recv_size, count))
                        sock.send(b"Server: OK")
                        count += 1
                        received_size += recv_size
                        fw.write(recv_file)
                    fw.close()

                print("Server: Received {} successfully".format(file_name))
                sock.send(b"OK")

            
            testImg = dict((imgName,Image.open(receive_path+'/'+imgName) ) for imgName in image_list)

            for imgName,img in testImg.items(): 
                resized = np.array(img.resize((227, 227))) - self.imgMean
                maxx = np.argmax(self.sess.run(self.softmax, feed_dict = {self.x: resized.reshape((1, 227, 227, 3))}))
                # maxx = 0 # for test use
                result = caffe_classes.class_names[maxx]
                print("{}: {}\n----".format(imgName,result))
                sock.send(bytes(imgName.encode('utf-8')))
                sock.recv(2)
                sock.send(bytes(result.encode('utf-8')))
            sock.close()
        else:
            print("Server: query invalid")
            return


if __name__ == '__main__':
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # ip = "192.168.1.199"
    # ip = "192.168.26.66"
    ip = "0.0.0.0"
    port = 50000
    sock.bind((ip, port))    
    sock.listen(1)
    server = Server()
    print("Waiting for connection...")
    while True:
        sock, addr = sock.accept()
        t = threading.Thread(target=server.run, args=(sock, addr))
        t.start()
        break