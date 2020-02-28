# 预测部署
## 预测模型导出
通常，PLSC在训练过程中保存的模型只包含模型的参数信息，而不包括预测模型结构。为了部署PLSC预测库，需要将预训练模型导出为预测模型。预测模型包括预测所需要的模型参数和模型结构，用于后续地预测任务(参见[预测库使用指南](#预测库使用指南))。

可以通过下面的代码将预训练模型导出为预测模型'export_for_inference.py'：

```python
from plsc import Entry

if __name__ == "__main__":
    ins = Entry()
    ins.set_checkpoint_dir('./pretrain_model')
    ins.set_model_save_dir('./inference_model')

    ins.convert_for_prediction()
```
其中'./pretrain_model'目录为预训练模型参数目录，'./inference_model'为用于预测的模型目录。

通过下面的命令行启动导出任务：
```shell script
python export_for_inference.py
```

## 预测库使用指南
python版本要求：
* python3
### 安装
#### server端安装

```shell script
pip3 install plsc-serving
```
#### client端安装

* 安装ujson:
```shell script
pip install ujson
```
* 复制[client脚本](./serving/client/face_service/face_service.py)到使用路径。

### 使用指南

#### server端使用指南

目前仅支持在GPU机器上进行预测，要求cuda版本>=9.0。

通过下面的脚本运行server端:

```python
from plsc_serving.run import PLSCServer
fs = PLSCServer()
#设定使用的模型文路径，str类型，绝对路径
fs.with_model(model_path = '/XXX/XXX')
#跑单个进程,gpu_index指定使用的gpu，int类型，默认为0；port指定使用的端口，int类型，默认为8866
fs.run(gpu_index = 0, port = 8010)
```

#### client端使用指南
通过下面的脚本运行client端:

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
fc.close()
```
