# Base64格式图像预处理

## 简介

实际业务中，一种常见的训练数据存储格式是将图像数据编码为base64格式。训练数据文件
的每一行存储一张图像的base64数据和该图像的标签，并通常以制表符('\t')分隔。

通常，所有训练数据文件的文件列表记录在一个单独的文件中，整个训练数据集的目录结构如下：

```shell
dataset
     |-- file_list.txt
     |-- dataset.part1
     |-- dataset.part2
     ...     ....
     `-- dataset.part10
```

其中，file_list.txt记录训练数据的文件列表，每行代表一个文件，以上面的例子来说，
file_list.txt的文件内容如下：

```shell
dataset.part1
dataset.part2
...
dataset.part10
```

而数据文件的每一行表示一张图像数据的base64表示，以及以制表符分隔的图像标签。

对于分布式训练，需要每张GPU卡处理相同数量的图像数据，并且通常需要在训练前做一次
训练数据的全局shuffle。

本文档介绍Base64格式图像预处理工具，用于在对训练数据做全局shuffle，并将训练数据均分到多个数据文件，
数据文件的数量和训练中使用的GPU卡数相同。当训练数据的总量不能整除GPU卡数时，通常会填充部分图像
数据（填充的图像数据随机选自训练数据集），以保证总的训练图像数量是GPU卡数的整数倍。

## 工具使用方法

工具位于tools目录下。
可以通过下面的命令行查看工具的使用帮助信息：

```python
python tools/process_base64_files.py --help
```

该工具支持以下命令行选项：

* data_dir: 训练数据的根目录
* file_list: 记录训练数据文件的列表文件，如file_list.txt
* nranks: 训练所使用的GPU卡的数量。

可以通过以下命令行运行该工具：

```shell
python tools/process_base64_files.py --data_dir=./dataset --file_list=file_list.txt --nranks=8
```

那么，会生成8个数量数据文件，每个文件中包含相同数量的训练数据。

最终的目录格式如下：

```shell
dataset
     |-- file_list.txt
     |-- dataset.part1
     |-- dataset.part2
     ...     ....
     `-- dataset.part8
```
