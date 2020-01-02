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
    * [分布式参数转换](#分布式参数转换)
* [设计思想](#设计思想)
    * [显存优化](#显存优化)
    * [通信优化](#通信优化)

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
    --cluster_node_ips="127.0.0.1" \
    --node_ip="127.0.0.1" \
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
### 预测库使用指南

## 高级功能
### 模型参数上传和下载(HDFS)
### Base64格式图像数据预处理
### 混合精度训练
### 分布式参数转换

## 设计思想

解决大规模分类问题的核心思想是采用模型并行方案实现深度神经网络模型的全连接层以及之后的损失值计算。

首先，我们回顾大规模分类问题面临的两个主要挑战：

1. 参数量过大，超出单个GPU卡的显存容量

2. 参数量较大，同步训练方式下通信开销较大

### 显存优化

为了解决显存不足的问题，PLSC采用模型并行设计，将深度神经网络的最后一层全连接层切分到各个GPU卡。全连接层天然地具有可切分属性，无外乎是一个矩阵乘法和加法（存在偏置项的情形下）。假设以100张GPU卡进行模型训练，当分类类别数目为一亿时，每张GPU卡上的全连接参数的大小约为2GB，这完全是可接受的。

对于全连接层计算，可以表示为矩阵乘法和加法，如下面的公示所示：

![FC计算公示](images/fc_computing.png)

其中，*W*和*b*全连接层参数，*X*是神经网络最后一层隐层的输出。将根据矩阵分块原理，全连接层计算又可以进一步地表示为下面的形式：

![FC计算公示展开](images/fc_computing_block.png)

这里，*n*是分块的块数。因此，我们可以将神经网络的最后一层全连接参数分布到多张GPU卡，并在每张卡上分别完成全连接层的部分计算，从而实现整个全连接层的计算，并解决大规模分类问题面临的GPU显存空间不足的问题。

需要注意的是，由于将神经网络模型最后一层全连接层参数划分到多张GPU卡，因此需要汇总各个GPU上的*X*参数，得到全连接层的全局输入*X*’(可以通过集合通信allgather实现)，并计算全连接层输出:

![全局FC计算公示](images/fc_computing_block_global.png)

### 通信优化

为了得到多分类任务的损失值，在完成全连接层计算后，通常会使用Softmax+交叉熵操作。

softmax的计算公示如下图所示：

![softmax计算公示](images/softmax_computing.png)

由于softmax计算是基于全类别的logit值的，因此需要进行全局同步，以计算分母项。这需要执行*N*次AllGather操作，这里*N*是参与训练的GPU卡数。这种全局通信方式的开销较大。

为了减少通信和计算代价，PLSC实现中仅同步其中的分母项。由于各个GPU卡上分母项是一个标量，所以可以显著降低通信开销。

## PLSC的特征：

- 基于源于产业实践的开源深度学习平台[飞桨平台](https://www.paddlepaddle.org.cn)

  飞桨是由百度研发的一款源于产业实践的开源深度学习平台，致力于让深度学习技术的创新与应用更简单。PLSC基于飞桨平台研发，实现与飞桨平台的无缝链接，可以更好地服务产业实践。

- 支持大规模分类

  单机8张V100 GPU配置下，支持的分类类别数增大了2.52倍;

- 包含多种预训练模型

  除了PLSC库源码，我们还发布了基于ResNet50模型、ResNet101模型、ResNet152模型的大规模分类模型在多种数据集上的预训练模型，方便用户基于这些预训练模型进行下游任务的fine-tuning。

- 提供从训练到部署的全流程解决方案

  PLSC库功能包括数据预处理、模型训练、验证和在线预测服务，提供从训练到部署的大规模分类问题全流程解决方案，用户可以基于PLSC库快速、便捷地搭建大规模分类问题解决方案。

## 预训练模型和性能

### 预训练模型

我们提供了下面的预训练模型，以帮助用户对下游任务进行fine-tuning。

| 模型             | 描述           |
| :--------------- | :------------- |
| [resnet50_distarcface_ms1m_arcface](https://plsc.bj.bcebos.com/pretrained_model/resnet50_distarcface_ms1mv2.tar.gz) | 该模型使用ResNet50网络训练，数据集为MS1M-ArcFace，训练阶段使用的loss_type为'dist_arcface'，预训练模型在lfw验证集上的验证精度为0.99817。 | 

### 训练性能

| 模型             | 训练集   | lfw  | agendb_30 | cfp_ff | cfp_fp |
| :--------------- | :------------- | :------ | :----- | :------ | :----  |
| ResNet50 | MS1M-ArcFace | 0.99817 | 0.99827 | 0.99857 | 0.96314 |
| ResNet50 | CASIA | 0.9895 | 0.9095 | 0.99057 | 0.915 |

备注：上述模型训练使用的loss_type为'dist_arcface'。更多关于ArcFace的内容请参考[ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)

### 基础功能
* [自定义模型](docs/custom_models.md)

### 预测部署

* [模型导出](docs/export_for_infer.md)
* [C++预测库使用](docs/serving.md)

### 高级功能

* [分布式参数转换](docs/distributed_params.md)
* [Base64格式图像预处理](docs/base64_preprocessor.md)
