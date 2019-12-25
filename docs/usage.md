# 模型训练和评估

PaddlePaddle大规模分类提供了从训练、评估到预测部署的全流程解决方案。本文档介绍如何使用PaddlePaddle大规模分类库快速完成训练、评估和预测部署。

## 数据准备

我们假设用户数据集的组织结构如下：

```shell
train_data/
|-- agedb_30.bin
|-- cfp_ff.bin
|-- cfp_fp.bin
|-- images
|-- label.txt
`-- lfw.bin
```

其中，*train_data*是用户数据的根目录，*agedb_30.bin*、*cfp_ff.bin*、*cfp_fp.bin*和*lfw.bin*分别是不同的验证数据集，且这些验证数据集不是全部必须的。本文档教程默认使用lfw.bin作为验证数据集，因此在浏览本教程时，请确保lfw.bin验证数据集可用。*images*目录包含JPEG格式的训练图像，*label.txt*中的每一行对应一张训练图像以及该图像的类别。

*label.txt*文件的内容示例如下：

```shell
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

## 模型训练

### 训练代码
下面的例子给出使用PLSC完成大规模分类训练的脚本*train.py*：
```python
from plsc import Entry

if __name__ == "__main__":
    ins = Entry()
    ins.train()
```

1. 从plsc包导入Entry类，其是使用PLCS大规模分类库功能的接口类。
2. 生成Entry类的实例。
3. 调用Entry类的train方法，即可开始训练。

默认地，训练阶段每个训练轮次的之后会使用验证集验证模型的效果，当没有验证数据集时，可以使用*set_with_test(False)* API关闭验证功能。

### 开始训练

下面的例子给出如何使用上述脚本启动训练任务：

```shell
python -m paddle.distributed.launch \
    --cluster_node_ips="127.0.0.1" \
    --node_ip="127.0.0.1" \
    --selected_gpus=0,1,2,3,4,5,6,7 \
    train.py
```

paddle.distributed.launch模块用于启动多机/多卡分布式训练任务脚本，简化分布式训练任务启动过程，各个参数的含义如下：

* cluster_node_ips: 参与训练的节点的ip地址列表，以逗号分隔；
* node_ip: 当前训练节点的ip地址；
* selected_gpus: 每个训练节点所使用的gpu设备列表，以逗号分隔。

## 模型验证

本教程中，我们使用lfw.bin验证数据集评估训练模型的效果。

### 验证代码

下面的例子给出使用PLSC完成大规模分类验证的脚本*val.py*：

```python
from plsc import Entry

if __name__ == "__main__":
    ins = Entry()
    ins.set_checkpoint("output/0")
    ins.test()
```

默认地，PLSC将训练脚本保存在'./ouput'目录下，并以pass作为区分不同训练轮次模型的子目录，例如'./output/0'目录下保存完成第一个轮次的训练后保存的模型。

在模型评估阶段，我们首先需要设置训练模型的目录，接着调用Entry类的test方法开始模型验证。
