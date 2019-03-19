import socket
import threading
import time
import hashlib
import struct
# import dill
import datetime
import os
import numpy as np
import tensorflow as tf
from apscheduler.schedulers.blocking import BlockingScheduler

BUFFER_SIZE = 1200
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

class sendScheduler(threading.Thread):
    def __init__(self):
        self._running = True
        self.scheduler = BlockingScheduler()
            
    def send_job(self, sock, send_file, times):
        print("[{}kB/s] Client: sending file packet {}...".format(len(send_file),times))
        sock.send(send_file)
        sock.recv(10)

    def run(self, sock, send_files):
        i = 0
        for send_file in send_files:
            self.scheduler.add_job(func=self.send_job, args=(sock, send_file, i, ), next_run_time=datetime.datetime.now() + datetime.timedelta(seconds=i))
            i += 1
        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            self.scheduler.shutdown()

    def terminate(self):
        self.scheduler.shutdown()
        print("Client: Send scheduler terminated")
        self._running = False
    
        

def client(ip, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("Client: Accept new connection from %s:%s..." % (ip, port))
    print("Client: Here is the client side...")
    
    # # Check the connection
    # sock.send(b"Client: Preparing to send the file to the server...")
    # isready = sock.recv(BUFFER_SIZE)
    # print(isready)
    try:
        sock.connect((ip, port))            
        print("Server: Connection Succeed")
    except Exception as e:
        print("Server: Connection ERROR")
        print("Exception: ", repr(e))
        exit()

    # Check the file to transfer
    isready = sock.recv(BUFFER_SIZE)
    print(isready)

    # image_list = [
    #     "000018.jpg",
    #     "002689.jpg",
    #     "005525.jpg"
    # ]

    #*********************Proping Processs********************
    isProp = True
    ip_list = [
        "192.168.1.128", # Raspberry
        "192.168.26.66" # Server
        "192.168.1.101", # Jetson
        "192.168.1.199", # Desktop
        "127.0.0.1", # Local
    ]
    # Assume all agree the port: 50000
    # for ip in ip_list:
    #     if isProp:
    #         target_ip, isProp = proping()
    #         if not isProp:
    #             break

    #*********************Transferring Process*******************
    image_path = "./testImages/"
    image_list = list(image_name for image_name in os.listdir(image_path))
    
    # times = 1
    # count = times
    # image_list *= times
    image_num = len(image_list)
    
    print("File num: ", image_num.to_bytes(4, byteorder='big'),"/",len(image_list))
    sock.send(image_num.to_bytes(4, byteorder='big'))
    # sumtime = 0

    with open("sprintGo.txt","r") as ulFile:
        size_arrange = list(int(float(ul)/8) for ul in ulFile.readlines())
        ulFile.close()
    # print("MIN:", min(size_arrange))
    # print("MAX:", max(size_arrange))

    for image_name in image_list:
        # print(count)
        # count -= 1
        file_size, md5 = get_file_info(image_path+image_name)
        file_info = struct.pack(HEAD_STRUCT, bytes(image_name.encode('utf-8')),len(image_name), file_size, md5)
        sock.send(file_info)
        receive_packet = sock.recv(BUFFER_SIZE)
        print(receive_packet)
        sent_size = 0

        require_size = size_arrange[0]
        # Send the file
        with open(image_path+image_name, 'rb') as img:
            # time0 = time.time()
            required_index = 0
            send_files = []
            while sent_size < file_size:
                remained_size = file_size - sent_size
                require_size = size_arrange[required_index]
                required_index += 1
                send_size = min(require_size, remained_size)
                send_files.append(img.read(send_size))
                sent_size += send_size
            img.close()
        print("Client: sending image {}...".format(image_name))
        send_scheduler = sendScheduler()
        send_thread = threading.Thread(target=send_scheduler.run, args=(sock,send_files,))
        send_thread.start()
        time.sleep(len(send_files))
        send_scheduler.terminate()
        print("Client: Send finished")

                # while current_remained_size > 0:
                #     send_size = BUFFER_SIZE if current_remained_size > BUFFER_SIZE else current_remained_size                
                #     send_files.append(img.read(send_size))
                #     sent_size += send_size
                #     current_remained_size -= send_size
                #     run_schedule(send_files)

                # sock.send(send_file)
                # rback = sock.recv(BUFFER_SIZE)
                # print(rback)
            # time1 = time.time()
            # sumtime += (time1 - time0)
            # img.close()
        # print("**************BAND:{}**************".format((time1-time0)))

        reply_packet = sock.recv(2)
        if reply_packet == b"OK":
            continue
        else:
            print("Connection ERROR.\nCurrent file:",image_name)
            break
    # print("**************BAND:{}**************".format((162.+16.1+14.8)*(times/sumtime)))

    # Receive the result from the server
    for i in range(image_num):
        imgName = sock.recv(BUFFER_SIZE)
        sock.send(b"OK")
        result = sock.recv(BUFFER_SIZE)
        print("{}: {}\n----".format(imgName,result))
    
    sock.close()
    return 


if __name__ == '__main__':
    # s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # ip = "0.0.0.0"
    # # ip = "127.0.0.1"
    # # ip = "192.168.26.66"
    # port = 50000
    # s.bind((ip, port))    
    # s.listen(1)
    # print("Client: Waiting for connection...")
    # while True:
    #     sock, addr = s.accept()
    #     t = threading.Thread(target=client, args=(sock, addr))
    #     t.start()
    #     # time.sleep(10)
    #     # print("Client: Stop the connection.")
    #     break
    print("Client: Test start...")
    ip = "127.0.0.1" # Local
    port = 50000
    client(ip, port)