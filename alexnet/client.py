import socket
import threading
import time
import hashlib
import struct
import datetime
import os
import numpy as np
import tensorflow as tf
from apscheduler.schedulers.blocking import BlockingScheduler
import random
import psutil
import math

BUFFER_SIZE = 1448
HEAD_STRUCT = '128sIqi32xs'
HEAD_INFO = 'fi'
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

    def get_file_info(self, file_path):
        # file_name = os.path.basename(file_path)
        # file_name_len = len(file_name)
        file_size = os.path.getsize(file_path)
        md5 = self.cal_md5(file_path)
        return file_size, bytes(md5.encode('utf-8'))

    def get_send_files(self, read_file):
        send_files = []
        sent_size = 0
        remained_size = len(read_file)
        while remained_size > BUFFER_SIZE:
            send_files.append(read_file[sent_size:sent_size+BUFFER_SIZE])
            sent_size += BUFFER_SIZE
            remained_size -= BUFFER_SIZE
        send_files.append(read_file[sent_size:sent_size+remained_size])
        return send_files

    def get_len_list(self, send_files_list):
        len_list = [len(send_files_list)]
        for send_files in send_files_list:
            len_list.append(len(send_files))
        for send_files in send_files_list:
            for send_file in send_files:
                len_list.append(len(send_file))
        # print(len(len_list))
        return len_list

###################
# SendScheduler: 
# Schedule send jobs with multi-thread. The file is pieced already before input. In
#  SendScheduler, one second sends one file so as to accurately control the 
#  bandwidth (kB/s). Note that it uses the thread termination to shutdown the send 
#  schedule.
###################
class SendScheduler(threading.Thread):
    def __init__(self):
        self._running = True
        self.scheduler = BlockingScheduler()
            
    def get_transfer_latency(self, send_files_list, band_list):
        sum_latency = len(band_list) - 1
        last_latency = len(send_files_list[-1])*BUFFER_SIZE/band_list[-1]
        return sum_latency + last_latency

    def send_job(self, sock, send_file, current_band, times):
        sock.send(send_file)
        print("[Client][{}kB/s] sending file packet {}...".format(current_band/1000,times), end = "")
        print(len(send_file))
        # print(send_file)
        sock.recv(12)

    def run(self, sock, send_files_list, band_list):
        if len(send_files_list) != len(band_list):
            print("[Send Scheduler] Send ERROR: The send files fail to match the bandwidth list.")
            return
        else:
            band_num = len(band_list)

        i = 0
        j = 0
        for i in range(band_num):
            current_band = band_list[i]
            packet_num = math.ceil(current_band/BUFFER_SIZE)
            if len(send_files_list[i]) != packet_num and i != band_num-1:
                print("[Send Scheduler]: Send ERROR: Wrong packet numbers.")
                print("i: {}|len: {}|packet_num:{}|band_num: {}".
                    format(i,len(send_files_list[i]),packet_num,band_num))
                return
            latency_delta = 1/packet_num*10
            send_files = send_files_list[i]
            for send_file in send_files:
                self.scheduler.add_job(func=self.send_job, args=(sock, send_file, current_band, j, ), 
                    next_run_time=datetime.datetime.now() + datetime.timedelta(seconds=latency_delta))
                j += 1
        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            self.scheduler.shutdown()        


    def terminate(self):
        self.scheduler.shutdown()
        print("[Client] Send scheduler terminated.")
        self._running = False
    

###################
# [Client] 
# The client. See notes on each function.
###################        
class Client(threading.Thread):
    def __init__(self):
        # self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        with open("sprintGo.txt","r") as ulFile:
            # self.size_arrange = list(int(float(ul)/8) for ul in ulFile.readlines())    # kB/s
            self.size_arrange = list(int(float(ul)/8*1000) for ul in ulFile.readlines()) # B/s
            ulFile.close()
        self.file_processor = FileProcessor()
        self.required_index = 0
        # Assume all agree the port: 50000
        self.port = 50000
        self.ip_list = [
            "192.168.1.128", # Raspberry
            "192.168.26.66" # Server
            "192.168.1.101", # Jetson
            "192.168.1.199", # Desktop
            "127.0.0.1", # Local
        ]
    
    ## Get local availabel resource
    def get_local_info(self):
        cpu_percent = psutil.cpu_percent(interval = None)
        conn_type = 0
        model_workload = 0.720 # 0.72 GFlops
        return cpu_percent, conn_type, model_workload


    ## Get current availabel ip list
    def set_ip_list(self, ip_list):
        self.ip_list = ip_list

    ## Get current availabel ip list
    def get_ip_list(self):
        ip_num = len(self.ip_list)
        remove_num = random.randint(0, ip_num-1)
        current_ip_list = self.ip_list
        for i in range(remove_num):
            current_num = len(current_ip_list)
            remove_index = random.randint(1,current_num-1)
            current_ip_list.pop(remove_index)
        return current_ip_list

    ## Porping each ip in ip_list
    def probing(self, ip, port = 50000):
        print("[Client] Start probing...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        isReady = self.check_conn(sock, ip, port)
        if not isReady:
            print("[Client] Probing connect Fail.")
            return
        probing_time0 = time.time()
        sock.send(b"[Client] Resource query..")
        ip_info = sock.recv(BUFFER_SIZE)
        probing_time1 = time.time()
        rtt = probing_time1 - probing_time0
        ip_cpu_percent, ip_conn_type = struct.unpack(HEAD_INFO,ip_info)
        print("[Client] Probing finished.")
        return ip_cpu_percent, ip_conn_type, self.size_arrange[self.required_index], rtt

    ## Run the client
    def run(self, ip, port = 50000):
        print("[Client] Here is the client side.")

        #*********************Probing Processs********************
        target_ip = ip
        ip_cpu_percent, ip_conn_type, curr_band, rtt = self.probing(target_ip)
        print("---\nCPU percent: {}\nConnection type: {}\nCurrent bandwidth: {}\nRTT: {}\n---"
            .format(ip_cpu_percent, ip_conn_type, curr_band, rtt))
        
        #*********************Transferring Process****************   
        self.transfer(target_ip, self.port)

    ## Check the connection
    def check_conn(self, sock, ip, port = 50000):
        # # Check the connection
        print("[Client] Connecting %s:%s..." % (ip, port))
        try:
            sock.connect((ip, port))            
            print("[Client] Connect Succeed.")
        except Exception as e:
            print("[Client] Connect ERROR.")
            print("Exception: ", repr(e))
            exit()
        # Check the file to transfer
        isReady = sock.recv(BUFFER_SIZE)
        # sock.send(b"[Client] Ready")
        if isReady.decode() == "[Server] Ready.":
            print(isReady)
            return True
        else:
            # print("Connection fail")
            return False


    ## Transfer the files
    def transfer(self, ip, port = 50000):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        isReady = self.check_conn(sock, ip, port)
        if not isReady:
            print("[Client] Transfer connect fail.")
            return
        sock.send(b"[Client] File transfer...")
        # sock.send(b"FW")
        # sock.close()
        # return
        
        image_path = "./testImages/"
        image_list = list(image_name for image_name in os.listdir(image_path))
        image_num = len(image_list)
        
        print("[Client] File num: ", image_num.to_bytes(4, byteorder='big'),"/",len(image_list))
        sock.send(image_num.to_bytes(4, byteorder='big'))
        # print("MIN:", min(self.size_arrange)) 106.84311625
        # print("MAX:", max(self.size_arrange)) 1183.70575

        # For each image, first send the file info. Then cut one file into file pieces
        # in send_files, which will then be input to the send scheduler to transfer 
        # according to the bandwidth dataset
        for image_name in image_list:
            file_processor = FileProcessor()
            file_size, md5 = file_processor.get_file_info(image_path+image_name)
            # print("File size: ", file_size)
            file_info = struct.pack(HEAD_STRUCT, bytes(image_name.encode('utf-8')),
                len(image_name), file_size, self.required_index ,md5)
            sock.send(file_info)
            receive_packet = sock.recv(BUFFER_SIZE)
            print(receive_packet)           

            # Cut the file to pieces
            sent_size = 0
            send_files_list = []
            band_list = []
            with open(image_path+image_name, 'rb') as img:                
                while sent_size < file_size:
                    remained_size = file_size - sent_size
                    current_band = self.size_arrange[self.required_index]
                    band_list.append(current_band)
                    self.required_index += 1
                    send_size = min(remained_size, current_band)
                    remained_size -= send_size
                    read_file = img.read(current_band)
                    send_files = self.file_processor.get_send_files(read_file)
                    send_files_list.append(send_files)
                    sent_size += send_size
                img.close()
            
            len_list = self.file_processor.get_len_list(send_files_list)
            sock.send(bytes(str(len_list).encode('utf-8')))

            # Run the send scheduler
            print("[Client] sending image {}...".format(image_name))
            send_scheduler = SendScheduler()
            send_thread = threading.Thread(target=send_scheduler.run, args=(sock,send_files_list,band_list))
            send_thread.start()
            transfer_latency = send_scheduler.get_transfer_latency(send_files_list, band_list)
            # print(transfer_latency)
            time.sleep(transfer_latency+1)
            send_scheduler.terminate()
            print("[Client] Send finished.")

            reply_packet = sock.recv(2)
            if reply_packet == b"OK":
                continue
            else:
                print("Connection ERROR.\nCurrent file: ",image_name)
                break

        # Receive the result from the server
        for i in range(image_num):
            # imgName = sock.recv(BUFFER_SIZE)
            imgResult = sock.recv(BUFFER_SIZE)
            imgName, result = imgResult.decode().split("||")
            sock.send(b"OK")
            # result = sock.recv(BUFFER_SIZE)
            print("{}: {}\n----".format(imgName,result))
        
        sock.close()


if __name__ == '__main__':
    print("[Client] Test start...")
    ip = "127.0.0.1" # Local
    # ip = "192.168.1.101" # Jetson
    # ip = "192.168.26.66" # Server
    port = 50000
    client = Client()
    client.run(ip, port)