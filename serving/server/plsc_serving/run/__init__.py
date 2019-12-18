# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import tarfile
import plsc_serving
import subprocess
import imp
import time


class PLSCServer():
    def __init__(self, with_gpu=True):
        os.chdir(self.get_path())
        self.with_gpu_flag = with_gpu
        self.p_list = []
        self.use_other_model = False
        self.run_m = False
        self.model_url = 'https://paddle-serving.bj.bcebos.com/paddle-gpu-serving/model-face'
        self.bin_url = 'https://paddle-serving.bj.bcebos.com/paddle-gpu-serving/bin-face'
        self.cpu_run_cmd = './bin/serving-cpu --bthread_min_concurrency=4 --bthread_concurrency=4 --logtostderr=true '
        self.gpu_run_cmd = './bin/serving-gpu --bthread_min_concurrency=4 --bthread_concurrency=4 --logtostderr=true '
        self.model_path_str = ''
        self.get_exe()

    def get_exe(self):
        exe_path = './bin'
        module_version = plsc_serving.__version__
        target_version_list = module_version.strip().split('.')
        target_version = target_version_list[0] + '.' + target_version_list[1]
        need_download = False

        if os.path.exists(exe_path):
            with open('./bin/serving-version.txt') as f:
                serving_version = f.read().strip()
            if serving_version != target_version:
                need_download = True
        else:
            need_download = True
        if need_download:
            tar_name = 'face-serving-' + target_version + '-bin.tar.gz'
            bin_url = self.bin_url + '/' + tar_name
            print('Frist time run, downloading PaddleServing components ...')
            os.system('wget ' + bin_url + ' --no-check-certificate')
            print('Decompressing files ..')
            tar = tarfile.open(tar_name)
            tar.extractall()
            tar.close()
            os.remove(tar_name)

    def modify_conf(self, gpu_index=0):
        os.chdir(self.get_path())
        engine_name = 'name: "face_resnet50"'
        if not self.with_gpu_flag:
            with open('./conf/model_toolkit.prototxt', 'r') as f:
                conf_str = f.read()
            conf_str = re.sub('GPU', 'CPU', conf_str)
            conf_str = re.sub('name.*"', engine_name, conf_str)
            conf_str = re.sub('model_data_path.*"', self.model_path_str,
                              conf_str)
            conf_str = re.sub('enable_memory_optimization: 0',
                              'enable_memory_optimization: 1', conf_str)
            open('./conf/model_toolkit.prototxt', 'w').write(conf_str)

        else:
            conf_file = './conf/model_toolkit.prototxt.' + str(gpu_index)
            with open(conf_file, 'r') as f:
                conf_str = f.read()
            conf_str = re.sub('CPU', 'GPU', conf_str)
            conf_str = re.sub('name.*"', engine_name, conf_str)
            conf_str = re.sub('model_data_path.*"', self.model_path_str,
                              conf_str)
            conf_str = re.sub('enable_memory_optimization: 0',
                              'enable_memory_optimization: 1', conf_str)
            open(conf_file, 'w').write(conf_str)

    def hold(self):
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("Server is going to quit")
            time.sleep(5)

    def run(self, gpu_index=0, port=8866):

        os.chdir(self.get_path())
        self.modify_conf(gpu_index)

        if self.with_gpu_flag == True:
            gpu_msg = '--gpuid=' + str(gpu_index) + ' '
            run_cmd = self.gpu_run_cmd + gpu_msg
            run_cmd += '--port=' + str(
                port) + ' ' + '--resource_file=resource.prototxt.' + str(
                    gpu_index) + ' '
            print('Start serving on gpu ' + str(gpu_index) + ' port = ' + str(
                port))
        else:
            re = subprocess.Popen(
                'cat /usr/local/cuda/version.txt > tmp 2>&1', shell=True)
            re.wait()
            if re.returncode == 0:
                run_cmd = self.gpu_run_cmd + '--port=' + str(port) + ' '
            else:
                run_cmd = self.cpu_run_cmd + '--port=' + str(port) + ' '
            print('Start serving on cpu port = {}'.format(port))

        process = subprocess.Popen(run_cmd, shell=True)

        self.p_list.append(process)
        if not self.run_m:
            self.hold()

    def run_multi(self, gpu_index_list=[], port_list=[]):
        self.run_m = True
        if len(port_list) < 1:
            print('Please set one port at least.')
            return -1
        if self.with_gpu_flag == True:
            if len(gpu_index_list) != len(port_list):
                print('Expect same length of gpu_index_list and port_list.')
                return -1
            for gpu_index, port in zip(gpu_index_list, port_list):
                self.run(gpu_index=gpu_index, port=port)
        else:
            for port in port_list:
                self.run(port=port)
        self.hold()

    def stop(self):
        for p in self.p_list:
            p.kill()

    def show_conf(self):
        '''
        with open('./conf/model_toolkit.prototxt', 'r') as f:
            conf_str = f.read()
        print(conf_str)
        '''

    def with_model(self, model_name=None, model_url=None):
        '''
        if model_url != None:
            self.mode_url = model_url
            self.use_other_model = True
        '''
        if model_name == None or type(model_name) != str:
            print('Please set model name string')
        os.chdir(self.get_path())
        self.get_model(model_name)

    def get_path(self):
        py_path = os.path.dirname(plsc_serving.__file__)
        server_path = os.path.join(py_path, 'server')
        return server_path

    def get_model(self, model_name):
        server_path = self.get_path()
        tar_name = model_name + '.tar.gz'
        model_url = self.model_url + '/' + tar_name

        model_path = os.path.join(server_path, 'data/model/paddle/fluid')
        if not os.path.exists(model_path):
            os.makedirs('data/model/paddle/fluid')
        os.chdir(model_path)
        if os.path.exists(model_name):
            pass
        else:
            os.system('wget ' + model_url + ' --no-check-certificate')
            print('Decompressing files ..')
            tar = tarfile.open(tar_name)
            tar.extractall()
            tar.close()
            os.remove(tar_name)

        self.model_path_str = r'model_data_path: "./data/model/paddle/fluid/' + model_name + r'"'
        os.chdir(self.get_path())
