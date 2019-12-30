# 混合精度训练

## 简介
PLSC支持混合精度训练。使用混合精度训练可以提升训练的速度，同时减少训练使用的显存开销。

## 使用方法
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
其中，`set_mixed_precision`函数介绍如下：

| API  | 描述 |
| --- | ---|
| set_mixed_precision| 设置混合精度训练

## 参数说明
set_mixed_precision 函数提供7个参数，其中use_fp16为必选项，决定是否开启混合精度训练，其他6个参数均有默认值，具体说明如下：

| 参数 | 类型 | 默认值| 说明
| --- | --- | ---|---|
|use_fp16|  bool | 无，需用户设定| 是否开启混合精度训练，设为True为开启混合精度训练
|init_loss_scaling| float | 1.0|初始的损失缩放值，这个值有可能会影响混合精度训练的精度，建议设为默认值
|incr_every_n_steps | int | 2000|累计迭代`incr_every_n_steps`步都没出现FP16的越界，loss_scaling则会增加`incr_ratio`倍，建议设为默认值
|decr_every_n_nan_or_inf| int | 2|累计迭代`decr_every_n_nan_or_inf`步出现了FP16的越界，loss_scaling则会缩小为原来的`decr_ratio`倍，建议设为默认值
|incr_ratio |float|2.0|扩大loss_scaling的倍数，建议设为默认值
|decr_ratio| float |0.5| 缩小loss_scaling的倍数，建议设为默认值
|use_dynamic_loss_scaling | bool | True| 是否使用动态损失缩放机制。如果开启，才会用到`incr_every_n_steps`，`decr_every_n_nan_or_inf`，`incr_ratio`，`decr_ratio`四个参数，开启会提高混合精度训练的稳定性和精度，建议设为默认值
|amp_lists|AutoMixedPrecisionLists类|None|自动混合精度列表类，可以指定具体使用fp16计算的operators列表，建议设为默认值


更多关于混合精度训练的介绍可参考：
- Paper: [MIXED PRECISION TRAINING](https://arxiv.org/abs/1710.03740)

- Nvidia Introduction: [Training With Mixed Precision](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html)

## 训练性能
配置： Nvidia Tesla v100 GPU 单机8卡

| 模型\速度 | FP32训练 | 混合精度训练 | 加速比 |
| --- | --- | --- | --- |
| ResNet50 | 2567.96 images/s | 3643.11 images/s | 1.42 |

备注：上述模型训练使用的loss_type均为'dist_arcface'。
