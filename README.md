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


