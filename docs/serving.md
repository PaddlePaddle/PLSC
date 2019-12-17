# PLSC Serving

### 安装

server端

需要python3环境，下载whl包

https://paddle-serving.bj.bcebos.com/paddle-gpu-serving/wheel/plsc_serving-0.1.4-py3-none-any.whl

pip3 install face_serving-0.1.3-py3-none-any.whl

client端

需要安装ujson，pip install ujson

复制[client脚本](./serving/client/face_service/face_service.py)到使用路径

### 使用

server端

目前仅支持在GPU机器上进行预测,运行环境要求cuda版本>=9.0。

```python
from plsc_serving.run import PLSCServer
fs = PLSCServer()
fs.with_model(model_name = 'face_resnet50')
#跑单个进程,gpu_index指定使用的gpu，int类型，默认为0；port指定使用的端口，int类型，默认为8866
fs.run(gpu_index = 0, port = 8010)
```

client端

```python
from face_service import FaceService
with open('./data/00000000.jpg', 'rb') as f:
    image = f.read()
fc = FaceService()
#添加server端连接，str类型，默认本机8010端口
fc.connect('127.0.0.1:8010')
#调用server端预测，输入为样本列表list类型，返回值为样本对应的embedding结果,list类型，shape为 batch size * embedding size
result = fc.encode([image])
print(result[0])
bc.close()
```
