# coding:utf-8
import sys
import time
from base64 import b64encode
import ujson
import os
from PIL import Image

_ver = sys.version_info
is_py2 = (_ver[0] == 2)
is_py3 = (_ver[0] == 3)

if is_py2:
    import httplib
if is_py3:
    import http.client as httplib


class FaceService():
    def __init__(self):
        self.con_list = []
        self.con_index = 0
        self.server_list = []

    def connect(self, server='127.0.0.1:8010'):
        self.server_list.append(server)
        con = httplib.HTTPConnection(server)
        self.con_list.append(con)

    def connect_all_server(self, server_list):
        for server in server_list:
            self.server_list.append(server)
            self.con_list.append(httplib.HTTPConnection(server))

    def infer(self, request_msg):

        try:
            cur_con = self.con_list[self.con_index]
            cur_con.request('POST', "/FaceClassifyService/inference",
                            request_msg, {"Content-Type": "application/json"})
            response = cur_con.getresponse()
            response_msg = response.read()
            #print(response_msg)
            response_msg = ujson.loads(response_msg)
            self.con_index += 1
            self.con_index = self.con_index % len(self.con_list)
            return response_msg

        except BaseException as err:
            del self.con_list[self.con_index]
            print(err)
            if len(self.con_list) == 0:
                print('All server failed')
                return 'fail'
            else:
                self.con_index = 0
                return 'retry'

    def encode(self, images):
        request = []
        for image in range(images):
            request.append(b64encode(image).decode('ascii'))

        #request
        request = {"base64_string": request}
        request_msg = ujson.dumps(request)

        response_msg = self.infer(request_msg)
        result = []
        for msg in response_msg["instance"]:
            result.append(msg["embedding"])

        #request end
        return result

    def close(self):
        for con in self.con_list:
            con.close()


def test():
    with open('./data/00000000.jpg', 'rb') as f:
        image = f.read()
    bc = FaceService()
    for i in image:
        print("%x" % i)
    bc.connect('127.0.0.1:8010')
    result = bc.encode([image])
    print(result[0])
    bc.close()


if __name__ == '__main__':
    test()
