# 分布式参数转换

## 简介

对于最后一层全连接层参数(W和b，假设参数b存在，否则，全连接参数仅为W)，通常切分到
所有训练GPU卡。那么，每个GPU卡上只保存部分全连接层参数。

当保存模型时，各个GPU卡的分布式参数均会得到保存。

在热启动或fine-tuning阶段，如果训练GPU卡数和热启动前或者预训练阶段使用的GPU卡数
不同时，需要对分布式参数进行转换，以保证分布式参数的数量和训练使用的GPU卡数相同。

默认地，当使用train()方法时，会自动进行分布式参数的转换。

## 工具使用方法

分布式参数转换工具也可以单独使用，可以通过下面的命令查看使用方法：

```shell
python -m plsc.utils.process_distfc_parameter --help
```

该工具支持以下命令行选项：

| 选项                    |    描述              |
| :---------------------- | :------------------- |
| name_feature            | 分布式参数的名称特征，用于识别分布式参数。默认的，分布式参数的名称前缀为dist@arcface@rank@rankid或者dist@softmax@rank@rankid。其中，rankid为表示GPU卡的id。默认地，name_feature的值为@rank@。用户通常不需要改变该参数的值 |
| pretrain_nranks         | 预训练阶段使用的GPU卡数 |
| nranks                  | 本次训练将使用的GPU卡数 |
| num_classes             | 分类类别的数目          |
| emb_dim                 | 倒数第二层全连接层的输出维度，不包含batch size |
| pretrained_model_dir    | 预训练模型的保存目录 |
| output_dir              | 转换后分布式参数的保存目录 |

通常，在预训练模型中包含meta.pickle文件，该文件记录预训练阶段使用的GPU卡数，分类类别书和倒数第二层全连接层的输出维度，因此通常不需要指定pretrain_nranks、num_classes和emb_dim参数。

可以通过以下命令转换分布式参数：
```shell
python -m plsc.utils.process_distfc_parameter --nranks=4 --pretrained_model_dir=./output --output_dir=./output_post
```

需要注意的是，转换后的分布式参数保存目录只包含转换后的分布式参数，而不包含其它模型参数。因此，通常需要使用转换后的分布式参数替换
预训练模型中的分布式参数。
