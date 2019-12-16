# 预测模型导出

通常，PaddlePaddle大规模分类库在训练过程中保存的模型只保存模型参数信息，
而不包括预测模型结构。为了部署PLSC预测库，需要将预训练模型导出为预测模型。

可以通过下面的代码将预训练模型导出为预测模型：

```python
import plsc.entry as entry

if __name__ == "__main__":
    ins = entry.Entry()
    ins.set_checkpoint_dir('./pretrain_model')
    ins.set_model_save_dir('./inference_model')

    ins.convert_for_prediction()
```
