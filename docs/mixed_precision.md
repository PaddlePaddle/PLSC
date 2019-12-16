# 混合精度训练

PLSC支持混合精度训练。使用混合精度训练可以提升训练的速度，同时减少训练使用的内存。

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
其中，`set_mixed_precision`函数为设置混合精度训练是否开启。
默认地，混合精度训练为False，`loss_scaling`值为1.0，动态损失scaling训练设置为True。
`loss_scaling`值为初始的损失缩放值，这个值有可能会影响混合精度训练的精度，建议先设置为默认值，后续根据训练情况调参。
动态损失scaling训练一般可以提高训练的稳定性和精度，因此建议选择默认值。
