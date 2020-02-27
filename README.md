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

## 快速开始
请参考[快速开始](docs/source/md/quick_start.md)获取安装指南和快速使用示例。

## 预测部署
请参考[预测部署指南](docs/source/md/serving.md)获取预测部署使用指南。

## 高阶功能
请参考[进阶指南](docs/source/md/advanced.md)获取更多高阶功能的使用指南，如HDFS文件系统的自动上传和下载等。

## API参考
请参考[API参考](docs/source/md/api_reference.md)获取API使用信息。

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
