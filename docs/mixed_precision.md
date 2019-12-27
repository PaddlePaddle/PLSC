# 混合精度训练

## 简介
PLSC支持混合精度训练。使用混合精度训练可以提升训练的速度，同时减少训练使用的显存开销。

## 使用方法
可以通过下面的代码设置开启混合精度训练：

```python
from __future__ import print_function
import plsc.entry as entry
def main():
    ins = entry.Entry()
    ins.set_mixed_precision(True, 1.0)
    ins.train()
if __name__ == "__main__":
    main()
```
其中，`set_mixed_precision`函数介绍如下：

| API  | 描述    | 参数说明  |
| :------------------- | :--------------------| :----------------------  |
| set_mixed_precision(use_fp16, loss_scaling) | 设置混合精度训练  | `use_fp16`为是否开启混合精度训练，默认为False；`loss_scaling`为初始的损失缩放值，默认为1.0|

- `use_fp16`：bool类型，当想要开启混合精度训练时，可将此参数设为True即可。
- `loss_scaling`：float类型，为初始的损失缩放值，这个值有可能会影响混合精度训练的精度，建议设为默认值1.0。

为了提高混合精度训练的稳定性和精度，默认开启了动态损失缩放机制。更多关于混合精度训练的介绍可参考：[混合精度训练](https://arxiv.org/abs/1710.03740)

## 训练性能

| 模型\速度(单机8卡) | 正常训练 | 混合精度训练 | 加速比 |
| --- | --- | --- | --- |
| ResNet50 | 2567.96 images/s | 3643.11 images/s | 1.42 |
备注：上述模型训练使用的loss_type均为'dist_arcface'。
