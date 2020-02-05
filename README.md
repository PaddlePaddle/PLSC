# PLSC: 飞桨大规模分类库

## 简介
深度学习中用于解决多分类问题的深度神经网络的最后一层通常是全连接层和Softmax的组合层，并采用交叉熵(Cross-Entropy)算法计算神经网络的损失函数。由于全连接层的参数量随着分类类别数的增长线性增长，当分类类别数相当大时，神经网络的训练会面临下面两个主要挑战：

1. 参数量过大，超出单个GPU卡的显存容量：假设分类网络最后一层隐层的输出维度为512，那么当分类类别数为一百万时，最后一层全连接层参数的大小约为2GB（假设以32比特浮点数表示参数）。当分类问题的类别数为一亿时（例如，对自然界中的生物进行分类），则最后一层全连接层参数的大小接近200GB，远远超过当前GPU的显存容量。

2. 参数量较大，同步训练方式下通信开销较大：数据并行训练方式下，所有GPU卡之间需要同步参数的梯度信息，以完成参数值的同步更新。当参数数量较大时，参数的梯度信息数据量同样较大，从而导致参数梯度信息的通信开销较大，影响训练速度。

飞桨大规模分类(PLSC: **P**addlePaddle **L**arge **S**cale **C**lassification)库是基于[飞桨平台](https://github.com/PaddlePaddle/Paddle)构建的超大规模分类库，为用户提供从训练到部署的大规模分类问题全流程解决方案。

## PLSC特性
* 支持超大规模分类：单机8张V100 GPU配置下支持的最大类别数扩大2.52倍，支持的类别数随GPU卡数的增加而增加；
* 训练速度快：单机8张V100 GPU配置下，基于ResNet50模型的百万类别分类训练速度2,122.56 images/s, 并支持多机分布式训练和混合精度训练;
* 支持训练训练卡数的调整：加载模型参数的热启动训练可以使用和预训练不同的GPU卡数，并自动进行参数转换；
* base64格式图像数据预处理：提供base64格式图像数据的预处理，包括数据的全局shuffle，数据自动切分；
* 支持自定义模型：PLSC内建ResNet50、ResNet101和ResNet152模型，并支持用户自定义模型；
* 支持模型参数在HDFS文件系统的自动上传和下载；
* 全流程解决方案：提供从训练到部署的大规模分类问题全流程解决方案。

## 文档目录
* [快速入门](#快速入门)
    * [安装说明](#安装说明)
    * [训练和验证](#训练和验证)
    * [API介绍](#API介绍)
* [预测部署](#预测部署)
    * [预测模型导出](#预测模型导出)
    * [预测库使用指南](#预测库使用指南)
* [高级功能](#高级功能)
    * [模型参数上传和下载(HDFS)](#模型参数上传和下载(HDFS))
    * [Base64格式图像数据预处理](#Base64格式图像数据预处理)
    * [混合精度训练](#混合精度训练)
    * [自定义模型](#自定义模型)
    * [自定义训练数据](#自定义训练数据)
* [预训练模型和性能](#预训练模型和性能)
    * [预训练模型](#预训练模型)
    * [训练性能](#训练性能)

## 快速入门
### 安装说明
Python版本要求：
* python 2.7+
#### 1. 安装PaddlePaddle
##### 1.1 版本要求：
* PaddlePaddle>=1.6.2或开发版

##### 1.2 pip安装

当前，需要在GPU版本的PaddlePaddle下使用大规模分类库。

```shell script
pip install paddlepaddle-gpu
```

关于PaddlePaddle对操作系统、CUDA、cuDNN等软件版本的兼容信息，以及更多PaddlePaddle的安装说明，请参考[PaddlePaddle安装说明](https://www.paddlepaddle.org.cn/documentation/docs/zh/beginners_guide/install/index_cn.html)。

#### 2. 安装依赖包

```shell script
pip install -r requirements.txt
```
直接使用requirement.txt安装依赖包默认会安装最新的稳定版本PaddlePaddle。如需要使用开发版本的PaddlePaddle，请先通过下面的命令行卸载已安装的PaddlePaddle，并重新安装开发版本的PaddlePaddle。关于如何安装获取开发版本的PaddlePaddle，请参考[多版本whl包列表](https://www.paddlepaddle.org.cn/documentation/docs/zh/beginners_guide/install/Tables.html#ciwhls)。

```shell script
pip uninstall paddlepaddle-gpu
```

#### 3. 安装PLSC大规模分类库

可以直接使用pip安装PLSC大规模分类库：

```shell script
pip install plsc
```

### 训练和验证

PLSC提供了从训练、评估到预测部署的全流程解决方案。本文档介绍如何使用PLSC快速完成模型训练和模型效果验证。

#### 数据准备

我们假设用户数据的组织结构如下：

```shell script
train_data/
|-- agedb_30.bin
|-- cfp_ff.bin
|-- cfp_fp.bin
|-- images
|-- label.txt
`-- lfw.bin
```

其中，*train_data*是用户数据的根目录，*agedb_30.bin*、*cfp_ff.bin*、*cfp_fp.bin*和*lfw.bin*分别是不同的验证数据集，且这些验证数据集不是必须的。本教程默认使用lfw.bin作为验证数据集，因此在浏览本教程时，请确保lfw.bin验证数据集可用。*images*目录包含JPEG格式的训练图像，*label.txt*中的每一行对应一张训练图像以及该图像的类别。

*label.txt*文件的内容示例如下：

```shell script
images/00000000.jpg 0
images/00000001.jpg 0
images/00000002.jpg 0
images/00000003.jpg 0
images/00000004.jpg 0
images/00000005.jpg 0
images/00000006.jpg 0
images/00000007.jpg 0
... ...
```
其中，每一行表示一张图像的路径和该图像对应的类别，图像路径和类别间以空格分隔。

#### 模型训练
##### 训练代码
下面给出使用PLSC完成大规模分类训练的脚本文件*train.py*：
```python
from plsc import Entry

if __name__ == "__main__":
    ins = Entry()
    ins.set_train_epochs(1)
    ins.set_model_save_dir("./saved_model")
    # ins.set_with_test(False)  # 当没有验证集时，请取消该行的注释
    # ins.set_loss_type('arcface')  # 当仅有一张GPU卡时，请取消该行的注释
    ins.train()
```
使用PLSC开始训练，包括以下几个主要步骤：
1. 从plsc包导入Entry类，该类是PLCS大规模分类库所有功能的接口类。
2. 生成Entry类的实例。
3. 调用Entry类的train方法，开始模型训练。

默认地，该训练脚本使用的loss值计算方法为'dist_arcface'，需要两张或以上的GPU卡，当仅有一张可用GPU卡时，可以使用下面的语句将loss值计算方法改为'arcface'。

```python
ins.set_loss_type('arcface')
```

默认地，训练过程会在每个训练轮次之后会使用验证集验证模型的效果，当没有验证数据集时，可以使用*set_with_test(False)*关闭模型验证。

##### 启动训练任务

下面的例子给出如何使用上述脚本启动训练任务：

```shell script
python -m paddle.distributed.launch \
    --cluster_node_ips="127.0.0.1" \
    --node_ip="127.0.0.1" \
    --selected_gpus=0,1,2,3,4,5,6,7 \
    train.py
```

paddle.distributed.launch模块用于启动多机/多卡分布式训练任务脚本，简化分布式训练任务启动过程，各个参数的含义如下：

* cluster_node_ips: 参与训练的机器的ip地址列表，以逗号分隔；
* node_ip: 当前训练机器的ip地址；
* selected_gpus: 每个训练节点所使用的gpu设备列表，以逗号分隔。

对于单机多卡训练任务，可以省略cluster_node_ips和node_ip两个参数，如下所示：

```shell script
python -m paddle.distributed.launch \
    --selected_gpus=0,1,2,3,4,5,6,7 \
    train.py
```

当仅使用一张GPU卡时，请使用下面的命令启动训练任务：
```shell script
python train.py
```

#### 模型验证

本教程中，我们使用lfw.bin验证集评估训练模型的效果。

##### 验证代码

下面的例子给出模型验证脚本*val.py*：

```python
from plsc import Entry

if __name__ == "__main__":
    ins = Entry()
    ins.set_checkpoint_dir("./saved_model/0/")
    ins.test()
```

训练过程中，我们将模型参数保存在'./saved_model'目录下，并将每个epoch的模型参数分别保存在不同的子目录下，例如'./saved_model/0'目录下保存的是第一个epoch训练完成后的模型参数，以此类推。

在模型验证阶段，我们首先需要设置模型参数的目录，接着调用Entry类的test方法开始模型验证。

##### 启动验证任务

下面的例子给出如何使用上述脚本启动验证任务：

```shell script
python -m paddle.distributed.launch \
    --selected_gpus=0,1,2,3,4,5,6,7 \
    val.py
```

使用上面的脚本，将在多张GPU卡上并行执行验证任务，缩短验证时间。

当仅有一张GPU卡可用时，可以使用下面的命令启动验证任务：
```shell script
python val.py
```

### API介绍

#### 默认配置信息

PLSC大规模提供了默认配置参数，用于设置模型训练、验证和模型参数等信息，如训练数据集目录、训练轮数等。

这些默认参数位于plsc.config中，下面给出这些参数的含义和默认值。

##### 模型训练相关信息

| 参数名称          | 参数含义            | 默认值                  |
| :-------         | :-------           | :-----                 |
| train_batch_size | 训练阶段batch size值 | 128                   |
| dataset_dir      | 数据集根目录         | './train_data'        |
| train_image_num  | 训练图像的数量       | 5822653               |
| train_epochs     | 训练轮数            | 120                   |
| warmup_epochs    | warmup的轮数        | 0                     |
| lr               | 初始学习率           | 0.1                   |
| lr_steps         | 学习率衰减的步数     | (100000,160000,220000) |

##### 模型验证相关信息

| 参数名称         | 参数含义                                | 默认值   |
| :-------        | :-------                               | :-----  |
| val_targets     | 验证数据集名称，以逗号分隔，如'lfw,cfp_fp' | lfw      |
| test_batch_size | 验证阶段batch size的值                   | 120     |
| with_test       | 是否在每轮训练之后开始验证模型效果          | True     |

##### 模型参数相关信息

| 参数名称        | 参数含义                            | 默认值                |
| :-------       | :-------                           | :-----               |
| model_name     | 使用的模型的名称                     | 'RestNet50'           |
| checkpoint_dir | 预训练模型（checkpoint）目录         | ""，表示不使用预训练模型 |
| model_save_dir | 训练模型的保存目录                   | "./output"            |
| loss_type      | loss计算方法，                      | 'dist_arcface'        |
| num_classes    | 分类类别的数量                       | 85742                |
| image_shape    | 图像尺寸列表，格式为CHW               | [3, 112, 112]        |
| margin         | dist_arcface和arcface的margin参数   | 0.5                   |
| scale          | dist_arcface和arcface的scale参数    | 64.0                  |
| emb_size       | 模型最后一层隐层的输出维度            | 512                   |

备注：

* checkpoint_dir和model_save_dir的区别：
    * checkpoint_dir指用于在训练/验证前加载的预训练模型所在目录；
    * model_save_dir指的是训练模型的保存目录。

#### 参数设置API

PLSC的Entry类提供了下面的API，用于修改默认参数信息：

* set_val_targets(targets)
    * 设置验证数据集名称，以逗号分隔，类型为字符串。
* set_train_batch_size(size)
    * 设置训练batch size的值，类型为int。
* set_test_batch_size(size)
    * 设置验证batch size的值，类型为int。
* set_mixed_precision(use_fp16,init_loss_scaling=1.0,incr_every_n_steps=2000,decr_every_n_nan_or_inf=2,incr_ratio=2.0,decr_ratio=0.5,use_dynamic_loss_scaling=True,amp_lists=None)
    * 设置是否使用混合精度训练，以及相关的参数，具体的参数含义请参考[混合精度训练](#混合精度训练)。
* set_hdfs_info(fs_name,fs_ugi,fs_dir_for_save,fs_checkpoint_dir)
    * 设置hdfs文件系统信息，具体参数含义如下：
        * fs_name: hdfs地址，类型为字符串；
        * fs_ugi: 逗号分隔的用户名和密码，类型为字符串；
        * fs_dir_for_save: 模型的上传目录，当设置该参数时，会在训练结束后自动将保存的模型参数上传到该目录；
        * fs_checkpoint_dir: hdfs上预训练模型参数的保存目录，当设置该参数和checkpoint目录后，会在训练开始前自动下载模型参数。
* set_model_save_dir(dir)
    * 设置模型保存路径model_save_dir，类型为字符串。
* set_dataset_dir(dir)
    * 设置数据集根目录dataset_dir，类型为字符串。
* set_train_image_num(num)
    *  设置训练图像的总数量，类型为int。
* set_calc_acc(calc)
    * 设置是否在训练时计算acc1和acc5值，类型为bool，在训练过程中计算acc值会占用额外的显存空间，导致支持的类别数下降，仅在必要时设置。
* set_class_num(num)
    * 设置分类类别的总数量，类型为int。
* set_emb_size(size)
    * 设置最后一层隐层的输出维度，类型为int。
* set_model(model)
    * 设置用户自定义模型类实例，BaseModel的子类的实例。
* set_train_epochs(num)
    * 设置训练的轮数，类型为int。
* set_checkpoint_dir(dir)
    * 设置用于加载的预训练模型的目录，类型为字符串。
* set_warmup_epochs(num)
    * 设置warmup的轮数，类型为int。
* set_loss_type(loss_type)
    * 设置模型loss值的计算方法，可选项为'arcface'，'softmax', 'dist_softmax'和'dist_arcface'，类型为字符串;
    * 'arcface'和'softmax'表示只使用数据并行，而不是用分布式FC参数，'distarcface'和'distsoftmax'表示使用分布式版本的arcface和softmax，即将最后一层FC的参数分布到多张GPU卡上；
    * 关于arcface的细节请参考[ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)。
* set_image_shape(size)
    * 设置图像尺寸，格式为CHW，类型为元组或列表。
* set_optimizer(optimizer)
    * 设置训练阶段的optimizer，值为Optimizer类或其子类的实例。
* set_with_test(with_test)
    * 设置是否在每完成一轮训练后验证模型效果，类型为bool。
* set_distfc_attr(param_attr=None, bias_attr=None)
    * 设置最后一层FC的W和b参数的属性信息，请参考[参数属性信息](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/fluid_cn/ParamAttr_cn.html#paramattr)。
* convert_for_prediction()
    * 将预训练模型转换为预测模型。
* test()
    * 模型验证。
* train()
    * 模型训练。

备注：set_warmup_epochs和set_image_num函数的附加说明：

默认的，我们认为训练过程中总的batch size为1024时可以取得较好的训练效果。例如，当使用8张GPU时，每张GPU卡上的batch size为128。当训练过程中总的batch size不等于1024时，需要根据batch size调整初始学习率的大小，即：lr = total_batch_size / 1024 * lr。这里，lr表示初始学习率大小。另外，为了保持训练的稳定性，通常需要设置warmup_epochs，即在最初的warmup_epochs轮训练中，学习率有一个较小的值逐步增长到初始学习率。为了实现warmup过程，我们需要根据训练数据集中图像的数量计算每轮的训练步数。

当需要改变这种逻辑设定时，可以自定义实现Optimizer，并通过set_optimizer函数设置。

本节介绍的API均为PLSC的Entry类的方法，需要通过该类的实例调用，例如：

```python
from plsc import Entry

ins = Entry()
ins.set_class_num(85742)
ins.train()
```

## 预测部署
### 预测模型导出
通常，PLSC在训练过程中保存的模型只包含模型的参数信息，而不包括预测模型结构。为了部署PLSC预测库，需要将预训练模型导出为预测模型。预测模型包括预测所需要的模型参数和模型结构，用于后续地预测任务(参见[预测库使用指南](#预测库使用指南)。

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

### 预测库使用指南
python版本要求：
* python3
#### 安装
##### server端安装

```shell script
pip3 install plsc-serving
```
##### client端安装

* 安装ujson:
```shell script
pip install ujson
```
* 复制[client脚本](./serving/client/face_service/face_service.py)到使用路径。

#### 使用指南

##### server端使用指南

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

##### client端使用指南
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
bc.close()
```

## 高级功能
### 模型参数上传和下载(HDFS)

当通过set_hdfs_info(fs_name,fs_ugi,fs_dir_for_save=None,fs_checkpoint_dir=None)函数设置了HDFS相关信息时，PLSC会在训练开始前自动下载云训练模型参数，并在训练结束后自动将保存的模型参数上传到HDFS指定目录。

#### 模型参数上传
使用模型参数上传的训练脚本示例如下：
```python
from plsc import Entry

if __name__ == "__main__":
    ins = Entry()
    ins.set_model_save_dir('./saved_model')
    ins.set_hdfs_info("you_hdfs_addr", "name,passwd", "some_dir")
    ins.train()
```
#### 模型参数下载
使用模型参数下载的训练脚本示例如下：
```python
from plsc import Entry

if __name__ == "__main__":
    ins = Entry()
    ins.set_checkpoint_dir('./saved_model')
    ins.set_hdfs_info("you_hdfs_addr",
                      "name,passwd",
                      fs_checkpoint_dir="some_dir")
    ins.train()
```
该脚本将HDFS系统中"some_dir"目录下的所有模型参数下载到本地"./saved_model"目录。请确保"./saved_model"目录存在。

### Base64格式图像数据预处理

实际业务中，一种常见的训练数据存储格式是将图像数据编码为base64格式存储，训练数据文件的每一行存储一张图像的base64数据和该图像的标签，并通常以制表符('\t')分隔图像数据和图像标签。

通常，所有训练数据文件的文件列表记录在一个单独的文件中，整个训练数据集的目录结构如下：

```shell script
dataset
     |-- file_list.txt
     |-- dataset.part1
     |-- dataset.part2
     ...     ....
     `-- dataset.part10
```

其中，file_list.txt记录训练数据的文件列表，每行代表一个文件，以上面的例子来讲，file_list.txt的文件内容如下：

```shell script
dataset.part1
dataset.part2
...
dataset.part10
```

而数据文件的每一行表示一张图像数据的base64表示，以及以制表符分隔的图像标签。

对于分布式训练，需要每张GPU卡处理相同数量的图像数据，并且通常需要在训练前做一次训练数据的全局shuffle。

本文档介绍Base64格式图像预处理工具，用于对训练数据做全局shuffle，并将训练数据均分到多个数据文件，数据文件的数量和训练中使用的GPU卡数相同。当训练数据的总量不能整除GPU卡数时，通常会填充部分图像数据（填充的图像数据随机选自训练数据集），以保证总的训练图像数量是GPU卡数的整数倍，即每个数据文件中包含相同数量的图像数据。

#### 使用指南

该工具位于tools目录下。使用该工具时，需要安装sqlite3模块。可以通过下面的命令安装：

```shell script
pip install sqlite3
```

可以通过下面的命令行查看工具的使用帮助信息：

```shell script
python tools/process_base64_files.py --help
```

该工具支持以下命令行选项：

* data_dir: 训练数据的根目录
* file_list: 记录训练数据文件的列表文件，如file_list.txt
* nranks: 训练所使用的GPU卡的数量。

可以通过以下命令行运行该工具：

```shell script
python tools/process_base64_files.py --data_dir=./dataset --file_list=file_list.txt --nranks=8
```

那么，会生成8个数据文件，每个文件中包含相同数量的训练数据。

可以使用plsc.utils.base64_reader读取base64格式图像数据。

### 混合精度训练
PLSC支持混合精度训练。使用混合精度训练可以提升训练的速度，同时减少训练使用的显存开销。
#### 使用指南
可以通过下面的代码设置开启混合精度训练：

```python
# for speed up
export FLAGS_sync_nccl_allreduce=1
export FLAGS_cudnn_exhaustive_search=0
export FLAGS_cudnn_batchnorm_spatial_persistent=1
export FLAGS_eager_delete_tensor_gb=0

from plsc import Entry

def main():
    ins = Entry()
    ins.set_mixed_precision(True)
    ins.train()
if __name__ == "__main__":
    main()
```

#### 参数说明
set_mixed_precision 函数提供7个参数，其中use_fp16为必选项，决定是否开启混合精度训练，其他6个参数均有默认值，具体说明如下：

| 参数 | 类型 | 默认值| 说明 |
| --- | --- | ---|---|
|use_fp16|  bool | 无，需用户设定| 是否开启混合精度训练，设为True为开启混合精度训练 |
|init_loss_scaling| float | 1.0|初始的损失缩放值，这个值有可能会影响混合精度训练的精度，建议设为默认值 |
|incr_every_n_steps | int | 2000|累计迭代`incr_every_n_steps`步都没出现FP16的越界，loss_scaling则会增加`incr_ratio`倍，建议设为默认值 |
|decr_every_n_nan_or_inf| int | 2|累计迭代`decr_every_n_nan_or_inf`步出现了FP16的越界，loss_scaling则会缩小为原来的`decr_ratio`倍，建议设为默认值 |
|incr_ratio |float|2.0|扩大loss_scaling的倍数，建议设为默认值 |
|decr_ratio| float |0.5| 缩小loss_scaling的倍数，建议设为默认值 |
|use_dynamic_loss_scaling | bool | True| 是否使用动态损失缩放机制。如果开启，才会用到`incr_every_n_steps`，`decr_every_n_nan_or_inf`，`incr_ratio`，`decr_ratio`四个参数，开启会提高混合精度训练的稳定性和精度，建议设为默认值 |
|amp_lists|AutoMixedPrecisionLists类|None|自动混合精度列表类，可以指定具体使用fp16计算的operators列表，建议设为默认值 |


更多关于混合精度训练的介绍可参考：
- Paper: [MIXED PRECISION TRAINING](https://arxiv.org/abs/1710.03740)

- Nvidia Introduction: [Training With Mixed Precision](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html)

#### 训练性能
配置： Nvidia Tesla v100 GPU 单机8卡

| 模型\速度 | FP32训练 | 混合精度训练 | 加速比 |
| --- | --- | --- | --- |
| ResNet50 | 2567.96 images/s | 3643.11 images/s | 1.42 |

备注：上述模型训练使用的loss_type均为'dist_arcface'。
### 自定义模型
默认地，PLSC构建基于ResNet50模型的训练模型。

PLSC提供了模型基类plsc.models.base_model.BaseModel，用户可以基于该基类构建自己的网络模型。用户自定义的模型类需要继承自该基类，并实现build_network方法，构建自定义模型。

下面的例子给出如何使用BaseModel基类定义用户自己的网络模型和使用方法：
```python
import paddle.fluid as fluid
from plsc import Entry
from plsc.models.base_model import BaseModel

class ResNet(BaseModel):
    def __init__(self, layers=50, emb_dim=512):
        super(ResNet, self).__init__()
        self.layers = layers
        self.emb_dim = emb_dim

    def build_network(self,
                      input,
                      label,
                      is_train):
        layers = self.layers
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, \
            "supported layers {}, but given {}".format(supported_layers, layers)

        if layers == 50:
            depth = [3, 4, 14, 3]
            num_filters = [64, 128, 256, 512]
        elif layers == 101:
            depth = [3, 4, 23, 3]
            num_filters = [256, 512, 1024, 2048]
        elif layers == 152:
            depth = [3, 8, 36, 3]
            num_filters = [256, 512, 1024, 2048]

        conv = self.conv_bn_layer(input=input,
                                  num_filters=64,
                                  filter_size=3,
                                  stride=1,
                                  pad=1,
                                  act='prelu',
                                  is_train=is_train)

        for block in range(len(depth)):
            for i in range(depth[block]):
                conv = self.bottleneck_block(
                    input=conv,
                    num_filters=num_filters[block],
                    stride=2 if i == 0 else 1,
                    is_train=is_train)

        bn = fluid.layers.batch_norm(input=conv,
                                     act=None,
                                     epsilon=2e-05,
                                     is_test=False if is_train else True)
        drop = fluid.layers.dropout(
            x=bn,
            dropout_prob=0.4,
            dropout_implementation='upscale_in_train',
            is_test=False if is_train else True)
        fc = fluid.layers.fc(
            input=drop,
            size=self.emb_dim,
            act=None,
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Xavier(uniform=False, fan_in=0.0)),
            bias_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.ConstantInitializer()))
        emb = fluid.layers.batch_norm(input=fc,
                                      act=None,
                                      epsilon=2e-05,
                                      is_test=False if is_train else True)
        return emb

	def conv_bn_layer(
        ... ...

if __name__ == "__main__":
    ins = Entry()
    ins.set_model(ResNet())
    ins.train()
```

用户自定义模型类需要继承自基类BaseModel，并实现build_network方法。

build_network方法的输入如下：
* input: 输入图像数据
* label: 图像类别
* is_train: 表示训练阶段还是测试/预测阶段

build_network方法返回用户自定义组网的输出变量。

### 自定义训练数据

默认地，我们假设用户的训练数据目录组织如下：

```shell script
train_data/
|-- images
`-- label.txt
```

其中，images目录中存放用户训练数据，label.txt文件记录用户训练数据中每幅图像的地址和对应的类别标签。

当用户的训练数据按照其它自定义格式组织时，可以按照下面的步骤使用自定义训练数据：

1. 定义reader函数(生成器），该函数对用户数据进行预处理（如裁剪），并使用yield生成数据样本；
    * 数据样本的格式为形如(data, label)的元组，其中data为解码和预处理后的图像数据，label为该图像的类别标签。
2. 使用paddle.batch封装reader生成器，得到新的生成器batched_reader；
3. 将batched_reader赋值给plsc.Entry类示例的train_reader成员。

为了便于描述，我们仍然假设用户训练数据组织结构如下：

```shell script
train_data/
|-- images
`-- label.txt
```

定义样本生成器的代码如下所示(reader.py)：

```python
import random
import os
from PIL import Image

def arc_train(data_dir):
    label_file = os.path.join(data_dir, 'label.txt')
    train_image_list = None
    with open(label_file, 'r') as f:
        train_image_list = f.readlines()
    train_image_list = get_train_image_list(data_dir)

    def reader():
        for j in range(len(train_image_list)):
            path, label = train_image_list[j]
            path = os.path.join(data_dir, path)
            img = Image.open(path)
            if random.randint(0, 1) == 1:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = np.array(img).astype('float32').transpose((2, 0, 1))
            yield img, label

    return reader
```

使用用户自定义训练数据的训练代码如下：

```python
import argparse
import paddle
from plsc import Entry
import reader

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",
                    type=str,
                    default="./data",
                    help="Directory for datasets.")
args = parser.parse_args()


def main():
    global args
    ins = Entry()
    ins.set_dataset_dir(args.data_dir)
    train_reader = reader.arc_train(args.data_dir)
    # Batch the above samples;
    batched_train_reader = paddle.batch(train_reader,
                                        ins.train_batch_size)
    # Set the reader to use during training to the above batch reader.
    ins.train_reader = batched_train_reader

    ins.train()


if __name__ == "__main__":
    main()
```

更多详情请参考[示例代码](./demo/custom_reader.py)

## 预训练模型和性能

### 预训练模型

我们提供了下面的预训练模型，以帮助用户对下游任务进行fine-tuning。

| 模型             | 描述           |
| :--------------- | :------------- |
| [resnet50_distarcface_ms1m_arcface](https://plsc.bj.bcebos.com/pretrained_model/resnet50_distarcface_ms1mv2.tar.gz) | 该模型使用ResNet50网络训练，数据集为MS1M-ArcFace，训练阶段使用的loss_type为'dist_arcface'，预训练模型在lfw验证集上的验证精度为0.99817。 | 

### 训练性能

| 模型             | 训练集          | lfw      | agendb_30 | cfp_ff   | cfp_fp |
| :--------------- | :------------- | :------ | :-----     | :------ | :----  |
| ResNet50         | MS1M-ArcFace   | 0.99817 | 0.99827    | 0.99857 | 0.96314 |
| ResNet50         | CASIA          | 0.9895  | 0.9095     | 0.99057 | 0.915 |

备注：上述模型训练使用的loss_type为'dist_arcface'。更多关于ArcFace的内容请参考[ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)
