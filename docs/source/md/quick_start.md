# 快速开始

## 安装说明

Python版本要求：
* python 2.7+

### 1. 安装PaddlePaddle
#### 1.1 版本要求：

```shell script
PaddlePaddle>=1.6.2或开发版
```

#### 1.2 pip安装
当前，需要在GPU版本的PaddlePaddle下使用大规模分类库。

```shell script
pip install paddlepaddle-gpu>=1.6.2
```

关于PaddlePaddle对操作系统、CUDA、cuDNN等软件版本的兼容信息，以及更多PaddlePaddle的安装说明，请参考[PaddlePaddle安装说明](https://www.paddlepaddle.org.cn/documentation/docs/zh/beginners_guide/install/index_cn.html)。

如需要使用开发版本的PaddlePaddle，请先通过下面的命令行卸载已安装的PaddlePaddle，并重新安装开发版本的PaddlePaddle。关于如何获取和安装开发版本的PaddlePaddle，请参考[多版本whl包列表](https://www.paddlepaddle.org.cn/documentation/docs/zh/beginners_guide/install/Tables.html#ciwhls)。

```shell script
pip uninstall paddlepaddle-gpu
```

### 2. 安装PLSC大规模分类库

可以直接使用pip安装PLSC大规模分类库：

```shell script
pip install plsc
```

## 训练和验证

PLSC提供了从训练、评估到预测部署的全流程解决方案。本节介绍如何使用PLSC快速完成模型训练和模型效果验证。

### 数据准备

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

### 模型训练
#### 训练代码
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

#### 启动训练任务

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

### 模型验证

本节我们使用lfw.bin验证集为例说明如何评估模型的效果。

#### 验证代码

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

#### 启动验证任务

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
