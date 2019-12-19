# PLSC API简介

## 默认配置参数

PLSC大规模分类库提供了默认配置参数，用于设置训练、评估和模型相关的信息，如训练数
据集目录、训练轮数等。

这些参数信息位于plsc.config模块中，下面给出这些参数的含义和默认值。

### 训练相关

| 参数名称 | 参数含义 | 默认值 |
| :------- | :------- | :----- |
| train_batch_size | 训练阶段batch size的值 | 128 |
| dataset_dir | 数据集根目录 | './train_data' |
| train_image_num | 训练图像的数量 | 5822653 |
| train_epochs | 训练轮数 | 120 |
| warmup_epochs | warmup轮数 | 0 |
| lr | 初始学习率 | 0.1 |
| lr_steps | 学习率衰减的步数 | (100000,160000,220000) |

### 评估相关

| 参数名称 | 参数含义 | 默认值 |
| :------- | :------- | :----- |
| val_targets | 验证数据集名称，以逗号分隔，如'lfw,cfp_fp' | lfw |
| test_batch_size | 评估阶段batch size的值 | 120 |
| with_test | 是否在每轮训练之后开始评估模型 | True |

### 模型相关

| 参数名称 | 参数含义 | 默认值 |
| :------- | :------- | :----- |
| model_name | 使用的模型的名称 | 'RestNet50' |
| checkpoint_dir | 预训练模型目录 | "" |
| model_save_dir | 训练模型的保存目录 | "./output" |
| loss_type | loss类型，可选值为softmax、arcface、dist_softmax和dist_arcface | 'dist_arcface' |
| num_classes | 分类类别的数量 | 85742 |
| image_shape | 图像尺寸列表，格式为CHW | [3, 112, 112] |
| margin | dist_arcface和arcface的margin参数 | 0.5 |
| scale | dist_arcface和arcface的scale参数 | 64.0 |
| emb_size | 模型最后一层隐层的输出维度 | 512 |

备注：

* checkpoint_dir和model_save_dir的区别：checkpoint_dir用于在训练/评估前加载的预训练模型所在目录；model_save_dir指的是训练后模型的保存目录。

### 参数设置API

可以通过该组API修改默认参数，具体API及其描述见下表。

| API                  | 描述                 | 参数说明                 |
| :------------------- | :--------------------| :----------------------  |
| set_val_targets(targets) | 设置验证数据集   | 以逗号分隔的验证集名称，类型为字符串 |
| set_train_batch_size(size) | 设置训练batch size的值 | 类型为int        |
| set_test_batch_size(size) | 设置评估batch size的值 | 类型为int         |
| set_hdfs_info(fs_name, fs_ugi, directory) | 设置hdfs文件系统信息 | fs_name为hdfs地址，类型为字符串；fs_ugi为逗号分隔的用户名和密码，类型为字符串；directory为hdfs上的路径 |
| set_model_save_dir(dir) | 设置模型保存路径model_save_dir | 类型为字符串 |
| set_dataset_dir(dir) | 设置数据集根目录dataset_dir | 类型为字符串 |
| set_train_image_num(num) | 设置训练图像的总数量 | 类型为int |
| set_calc_acc(calc) | 设置是否在训练是计算acc1和acc5值 | 类型为bool |
| set_class_num(num) | 设置分类类别的总数量 | 类型为int |
| set_emb_size(size) | 设置最后一层隐层的输出维度 | 类型为int |
| set_model(model) | 设置用户使用的自定义模型类实例 | BaseModel的子类 |
| set_train_epochs(num) | 设置训练的轮数 | 类型为int |
| set_checkpoint_dir(dir) | 设置用于加载的预训练模型的目录 | 类型为字符串 |
| set_warmup_epochs(num) | 设置warmup的轮数 | 类型为int |
| set_loss_type(loss_type) | 设置模型的loss类型 | 类型为字符串 |
| set_image_shape(size) | 设置图像尺寸，格式为CHW | 类型为元组 |
| set_optimizer(optimizer) | 设置训练阶段的optimizer | Optimizer类实例 |
| convert_for_prediction() | 将预训练模型转换为预测模型 | None |
| test() | 模型评估 | None |
| train() | 模型训练 | None |

备注：

当设置set_calc_acc的参数值为True，会在训练是计算acc1和acc5的值，但这回占用额外的显存空间。

上述API均为PaddlePaddle大规模分类库PLSC的Entry类的方法，需要通过该类的实例
调用，例如：

```python
from plsc import Entry

ins = Entry()
ins.set_class_num(85742)
ins.train()
```
