# 高级功能
## 模型参数上传和下载(HDFS)

当通过set_hdfs_info(fs_name,fs_ugi,fs_dir_for_save=None,fs_checkpoint_dir=None)函数设置了HDFS相关信息时，PLSC会在训练开始前自动下载云训练模型参数，并在训练结束后自动将保存的模型参数上传到HDFS指定目录。

### 模型参数上传
使用模型参数上传的训练脚本示例如下：
```python
from plsc import Entry

if __name__ == "__main__":
    ins = Entry()
    ins.set_model_save_dir('./saved_model')
    ins.set_hdfs_info("you_hdfs_addr", "name,passwd", "some_dir")
    ins.train()
```
### 模型参数下载
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

## Base64格式图像数据预处理

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

### 使用指南

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

## 混合精度训练
PLSC支持混合精度训练。使用混合精度训练可以提升训练的速度，同时减少训练使用的显存开销。
### 使用指南
可以通过下面的代码设置开启混合精度训练：

```python
from plsc import Entry

def main():
    ins = Entry()
    ins.set_mixed_precision(True)
    ins.train()
if __name__ == "__main__":
    main()
```

### 参数说明
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

### 训练性能
配置： Nvidia Tesla v100 GPU 单机8卡

| 模型\速度 | FP32训练 | 混合精度训练 | 加速比 |
| --- | --- | --- | --- |
| ResNet50 | 2567.96 images/s | 3643.11 images/s | 1.42 |

备注：上述模型训练使用的loss_type均为'dist_arcface'。
## 自定义模型
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

## 自定义训练数据

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

更多详情请参考[示例代码](../../../demo/custom_reader.py)
