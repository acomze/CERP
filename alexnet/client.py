import socket
import threading
import time
import hashlib
import struct
# import dill
import os
import numpy as np
import tensorflow as tf

BUFFER_SIZE = 1024
HEAD_STRUCT = '128sIq32s'
info_size = struct.calcsize(HEAD_STRUCT)

def cal_md5(file_path):
    with open(file_path, 'rb') as fr:
        md5 = hashlib.md5()
        md5.update(fr.read())
        md5 = md5.hexdigest()
        return md5

def get_file_info(file_path):
    # file_name = os.path.basename(file_path)
    # file_name_len = len(file_name)
    file_size = os.path.getsize(file_path)
    md5 = cal_md5(file_path)
    return file_size, bytes(md5.encode('utf-8'))

def client(sock, addr):
    print("Client: Accept new connection from %s:%s..." % addr)
    print("Client: Here is the client side...")
    
    # Check the connection
    sock.send(b"Client: Preparing to send the file to the server...")
    isready = sock.recv(BUFFER_SIZE)
    print(isready)

    # Check the file to transfer
    image_list = [
        "000018.jpg",
        "002689.jpg",
        "005525.jpg"
    ]
    image_path = "./testModel/"
    image_num = len(image_list)
    # print("File num: ",len(image_list).to_bytes(4, byteorder='big'),"/",len(image_list))
    sock.send(image_num.to_bytes(4, byteorder='big'))
    for image_name in image_list:
        file_size, md5 = get_file_info(image_path+image_name)
        file_info = struct.pack(HEAD_STRUCT, bytes(image_name.encode('utf-8')),len(image_name), file_size, md5)
        sock.send(file_info)
        receive_packet = sock.recv(BUFFER_SIZE)
        print(receive_packet)
        sent_size = 0

        # Send the file
        with open(image_path+image_name, 'rb') as img:
            time0 = time.time()
            while sent_size < file_size:
                remained_size = file_size - sent_size
                send_size = BUFFER_SIZE if remained_size > BUFFER_SIZE else remained_size
                send_file = img.read(send_size)
                sent_size += send_size
                sock.send(send_file)
                rback = sock.recv(BUFFER_SIZE)
                # print(rback)
            img.close()
            time1 = time.time()
            print("**************TIME:{}**************".format(time1-time0))

        reply_packet = sock.recv(2)
        if reply_packet == b"OK":
            continue
        else:
            print("Connection ERROR.\nCurrent file:",image_name)
            break


    # Receive the result from the server
    for i in range(image_num):
        imgName = sock.recv(BUFFER_SIZE)
        sock.send(b"OK")
        result = sock.recv(BUFFER_SIZE)
        print("{}: {}\n----".format(imgName,result))
    
    sock.close()
    return 


if __name__ == '__main__':
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ip = "0.0.0.0"
    # ip = "127.0.0.1"
    # ip = "192.168.26.66"
    port = 50000
    s.bind((ip, port))    
    s.listen(1)
    print("Client: Waiting for connection...")
    while True:
        sock, addr = s.accept()
        t = threading.Thread(target=client, args=(sock, addr))
        t.start()
        time.sleep(3)
        print("Client: Stop the connection.")
        break
        