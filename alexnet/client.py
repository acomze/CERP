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
    file_name = os.path.basename(file_path)
    file_name_len = len(file_name)
    file_size = os.path.getsize(file_path)
    md5 = cal_md5(file_path)
    return bytes(file_name.encode('utf-8')), file_name_len, file_size, bytes(md5.encode('utf-8'))

def client(sock, addr):
    print("Client: Accept new connection from %s:%s..." % addr)
    print("Client: Here is the client side...")
    
    # Check the connection
    sock.send(b"Client: Preparing to send the file to the server...")
    isready = sock.recv(BUFFER_SIZE)
    print(isready)

    # Check the file to transfer
    file_name, file_name_len, file_size, md5 = get_file_info("testModel/002689.jpg")
    file_info = struct.pack(HEAD_STRUCT, file_name, file_name_len, file_size, md5)
    sock.send(file_info)
    receive_packet = sock.recv(BUFFER_SIZE)
    print(receive_packet)
    sent_size = 0

    # Send the file
    with open("testModel/002689.jpg", 'rb') as fr:
        while sent_size < file_size:
            remained_size = file_size - sent_size
            send_size = BUFFER_SIZE if remained_size > BUFFER_SIZE else remained_size
            send_file = fr.read(send_size)
            sent_size += send_size
            sock.send(send_file)
            rback = sock.recv(BUFFER_SIZE)
            print(rback)
        fr.close()

    # Receive the result from the server
    result = sock.recv(BUFFER_SIZE)
    print(result)
    sock.close()



if __name__ == '__main__':
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ip = "0.0.0.0"
    # ip = "192.168.26.66"
    port = 50000
    s.bind((ip, port))    
    s.listen(1)
    print("Waiting for connection...")
    while True:
        sock, addr = s.accept()
        t = threading.Thread(target=client, args=(sock, addr))
        t.start()