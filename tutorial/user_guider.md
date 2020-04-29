# PLSC使用指南

本指南介绍使用飞桨大规模分类库PLSC，包括数据的组织、启动训练和预测验证。

## 数据准备

本节以MS1M-Arcface数据集为例进行说明。

首先，请下载MS1M-Arcface数据集：[数据集地址](MS1M-Arcface数据集)

解压后，数据的组织结构如下所示：

```shell
train_data/
|-- agedb_30.bin
|-- cfp_ff.bin
|-- cfp_fp.bin
|-- images
|-- label.txt
`-- lfw.bin
```

其中，agedb_30.bin、cfp_ff.bin、cfp_fp.bin和lfw.bin文件为验证数据集文件。images目录中存放训练数据集，格式为jpg格式图像。label.txt文件记录数据图像及其标签的对应格式。

label.txt文件的内容如下所示：

```shell
images/00000000.jpg 0
images/00000001.jpg 0
images/00000002.jpg 0
images/00000003.jpg 0
images/00000004.jpg 0
images/00000005.jpg 0
images/00000006.jpg 0
images/00000007.jpg 0
```

label.txt文件中，每行包含一张训练图像的相对路径和该图像对应的标签（类别），两部分之间由空格分隔。

如果用户需要使用自定义格式的数据集，请参考:[自定义训练数据](../docs/source/md/advanced.md)

## 模型训练

### 训练代码

下面给出使用PLSC完成大规模分类训练的脚本文件train.py：

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

### 启动训练任务

下面的例子给出如何使用上述脚本启动训练任务：

```python
python -m paddle.distributed.launch \
    --cluster_node_ips="127.0.0.1" \
    --node_ip="127.0.0.1" \
    --selected_gpus=0,1,2,3,4,5,6,7 \
    train.py
```

paddle.distributed.launch模块用于启动多机/多卡分布式训练任务脚本，简化分布式训练任务启动过程，各个参数的含义如下：

- cluster_node_ips: 参与训练的机器的ip地址列表，以逗号分隔；
- node_ip: 当前训练机器的ip地址；
- selected_gpus: 每个训练节点所使用的gpu设备列表，以逗号分隔。

对于单机多卡训练任务，可以省略cluster_node_ips和node_ip两个参数，如下所示：

```shell
python -m paddle.distributed.launch \
    --selected_gpus=0,1,2,3,4,5,6,7 \
    train.py
```

当仅使用一张GPU卡时，请使用下面的命令启动训练任务：

```shell
python train.py
```

### 模型验证

本节我们使用lfw.bin验证集为例说明如何评估模型的效果。

### 验证代码

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

### 启动验证代码

下面的例子给出如何使用上述脚本启动验证任务：

```shell
python -m paddle.distributed.launch \
    --selected_gpus=0,1,2,3,4,5,6,7 \
    val.py
```

使用上面的脚本，将在多张GPU卡上并行执行验证任务，缩短验证时间。

当仅有一张GPU卡可用时，可以使用下面的命令启动验证任务：

```shell
python val.py
```

