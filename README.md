# PLSC: PaddlePaddle大规模分类库

## 简介
PaddlePaddle大规模分类库(PLSC: PaddlePaddle Large Scale Classification)是
基于[飞桨平台](https://www.paddlepaddle.org.cn)构建的超大规模分类库，为用
户提供从训练到部署的大规模分类问题全流程解决方案。

深度学习中用于解决多分类问题的深度神经网络的最后一层通常是全连接层+Softmax，
并采用交叉熵(Cross-Entropy)算法计算网络的损失函数。由于全连接层的参数量随着
分类类别数的增长线性增长，当分类类别数相当大时，会面临下面两个主要挑战：

1. 参数量过大，超出单个GPU卡的显存容量：假设分类网络最后一层隐层的输出维度为512，
那么当分类类别数为一百万时，最后一层全连接层参数的大小约为2GB（假设以32比特浮点
数表示参数）。当分类问题的类别数为一亿时（例如，对自然界中的生物进行分类），则
最后一层全连接层参数的大小接近200GB，远远超过当前GPU的显存容量。

2. 参数量较大，同步训练方式下通信开销较大：数据并行训练方式下，所有GPU卡之间需
要同步参数的梯度信息，以完成参数值的同步更新。当参数数量较大时，参数的梯度信息
数据量同样较大，从而导致参数梯度信息的通信开销较大，影响训练速度。

为了解决大规模分类问题，我们设计开发了PaddlePaddle大规模分类库PLCS，为用户提供
从训练到部署的大规模分类问题全流程解决方案。

## 设计思想

解决大规模分类问题的核心思想是采用模型并行方案实现深度神经网络模型的全连接层以
及之后的损失值计算。

首先，我们回顾大规模分类问题面临的两个主要挑战：

1. 参数量过大，超出单个GPU卡的显存容量

2. 参数量较大，同步训练方式下通信开销较大

### 显存优化

为了解决显存不足的问题，PLSC采用模型并行设计，将深度神经网络的最后一层全连接层切
分到各个GPU卡。全连接层天然地具有可切分属性，无外乎是一个矩阵乘法和加法（存在偏置
项的情形下）。假设以100张GPU卡进行模型训练，当分类类别数目为一亿时，每张GPU卡上的
全连接参数的大小约为2GB，这完全是可接受的。

对于全连接层计算，可以表示为矩阵乘法和加法，如下面的公示所示：

![FC计算公示](images/fc_computing.png)

其中，*W*和*b*全连接层参数，*X*是神经网络最后一层隐层的输出。将根据矩阵分块原理，全
连接层计算又可以进一步地表示为下面的形式：

![FC计算公示展开](images/fc_computing_block.png)

这里，*n*是分块的块数。因此，我们可以将神经网络的最后一层全连接参数分布到多张GPU卡，
并在每张卡上分别完成全连接层的部分计算，从而实现整个全连接层的计算，并解决大规模分
类问题面临的GPU显存空间不足的问题。

需要注意的是，由于将神经网络模型最后一层全连接层参数划分到多张GPU卡，因此需要汇总
各个GPU上的*X*参数，得到全连接层的全局输入*X*’(可以通过集合通信allgather实现)，并
计算全连接层输出:

![全局FC计算公示](images/fc_computing_block_global.png)

### 通信优化

为了得到多分类任务的损失值，在完成全连接层计算后，通常会使用Softmax+交叉熵操作。

softmax的计算公示如下图所示：

![softmax计算公示](images/softmax_computing.png)

由于softmax计算是基于全类别的logit值的，因此需要进行全局同步，以计算分母项。这需
要执行*N*次AllGather操作，这里*N*是参与训练的GPU卡数。这种全局通信方式的开销较大。

为了减少通信和计算代价，PLSC实现中仅同步其中的分母项。由于各个GPU卡上分母项是一个
标量，所以可以显著降低通信开销。

## PLSC的特征：

- 基于源于产业实践的开源深度学习平台[飞桨平台](https://www.paddlepaddle.org.cn)

  飞桨是由百度研发的一款源于产业实践的开源深度学习平台，致力于让深度学习技术的创
	新与应用更简单。PLSC基于飞桨平台研发，实现与飞桨平台的无缝链接，可以更好地服务
	产业实践。

- 包含多种预训练模型
  除了PLSC库源码，我们还发布了基于ResNet50模型、ResNet101模型、ResNet152模型的大
	规模分类模型在多种数据集上的预训练模型，方便用户基于这些预训练模型进行下游任务
	的fine-tuning。

- 提供从训练到部署的全流程解决方案
  PLSC库功能包括数据预处理、模型训练、验证和在线预测服务，提供从训练到部署的大规
	模分类问题全流程解决方案，用户可以基于PLSC库快速、便捷地搭建大规模分类问题解决
	方案。

## 预训练模型和性能

### 预训练模型

我们提供了下面的预训练模型，以帮助用户对下游任务进行fine-tuning。

| 模型             | 描述           |
| :--------------- | :------------- |
| [resnet50_distarcface_ms1m_v2](http://icm.baidu-int.com/user-center/account) | 该模型使用ResNet50网络训练，数据集为MS1M_v2，训练阶段使用的loss_type为'dist_arcface'，预训练模型在lfw验证集上的验证精度为0.99817。 | 

### 训练性能

| 模型             | 训练集   | lfw  | agendb_30 | cfp_ff | cfp_fp |
| :--------------- | :------------- | :------ | :----- | :------ | :----  |
| ResNet50 | MS1M-ArcFace | 0.99817 | 0.99827 | 0.99857 | 0.96314 |
| ResNet50 | CASIA | 0.9895 | 0.9095 | 0.99057 | 0.915 |

备注：上述模型训练使用的loss_type为'dist_arcface'。更多关于ArcFace的内容请
参考[ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)

## 使用教程

我们提供了一系列使用教程，来帮助用户完成使用PLSC大规模分类库进行训练、评估和部署。

这一系列文档分为**快速入门**、**基础功能**、**预测部署**和**高级功能**四个部分，
由浅入深地介绍PLSC大规模分类库的使用方法。

### 快速入门

* [安装说明](docs/installation.md)
* [训练和验证](docs/usage.md)

### 基础功能

* [API简介](docs/api_intro.md)
* [自定义模型](docs/custom_models.md)

### 预测部署

* [模型导出](docs/export_for_infer.md)
* [C++预测库使用](docs/serving.md)

### 高级功能

* [分布式参数转换](docs/distributed_params.md)
* [Base64格式图像预处理](docs/base64_preprocessor.md)
