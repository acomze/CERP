import socket
import os
import sys
import struct
import pickle as P
import numpy as np
import argparse
import sys
import cv2
import tensorflow as tf
import numpy as np
import time

BUFFER_SIZE = 1024


# if args.mode == 'folder':
#     #get testImage
#     withPath = lambda f: '{}/{}'.format(args.path,f)
#     testImg = dict((f,cv2.imread(withPath(f))) for f in os.listdir(args.path) if os.path.isfile(withPath(f)))
# elif args.mode == 'url':
#     def url2img(url):
#         '''url to image'''
#         resp = urllib.request.urlopen(url)
#         image = np.asarray(bytearray(resp.read()), dtype="uint8")
#         image = cv2.imdecode(image, cv2.IMREAD_COLOR)
#         return image
#     testImg = {args.path:url2img(args.path)}




def socket_client():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('127.0.0.1', 50000))
    except socket.error as msg:
        print (msg)
        sys.exit(1)

    print (s.recv(1024))

    while True:
        for p, _ds, fs in os.walk(args.path): break
        bytes_list = [open(os.path.join(p, f), "rb").read() for f in fs]
        bytes = P.dumps(bytes_list)
        print(len(bytes))

        if len(bytes):
        # if testImg.values():
            # 定义定义文件信息。128s表示文件名为128bytes长，l表示一个int或log文件类型，在此为文件大小
            # 定义文件头信息，包含文件名和文件大小
            # arr = np.random.randint(0, 1000, size=(2, 3))
            # print(testImg)
            # s.send(P.dumps(open(filepath, "rb").read()))
            # s.send(P.dumps(testImg))

            # s.send(bytes)
            
            file_size = len(bytes)
            sent_size = 0
            while sent_size < file_size:
                remained_size = file_size - sent_size
                send_size = BUFFER_SIZE if remained_size > BUFFER_SIZE else remained_size
                send_file = bytes.read(send_size)
                sent_size += send_size
                s.send(send_file)
                rback = s.recv(2)

            # print('client filepath: {0}'.format(filepath))
            # print(filesize2)

            # result = s.recv(10240000)
            result = b""
            while True:
                resultPacket = s.recv(1024)
                if not resultPacket: break
                result += resultPacket

            bytes_list = P.loads(result)
            print(bytes_list)

        s.close()
        break


if __name__ == '__main__':
    start_time = time.time()
    socket_client()
    end_time = time.time()
    print('total time', end_time - start_time)
