# 预测模型导出

通常，PaddlePaddle大规模分类库在训练过程中保存的模型只保存模型参数信息，而不包括预测模型结构。为了部署PLSC预测库，需要将预训练模型导出为预测模型。预测模型包括预测所需要的模型参数和模型结构，用于后续地预测任务(参见[C++预测库使用](./serving.md)

预测模型包括预测所需要的模型参数和模型结构，用于后续地预测任务(参见[C++预测库使用](./serving.md))。

可以通过下面的代码将预训练模型导出为预测模型：

```python
from plsc import Entry

if __name__ == "__main__":
    ins = Entry()
    ins.set_checkpoint_dir('./pretrain_model')
    ins.set_model_save_dir('./inference_model')

    ins.convert_for_prediction()
```

其中'./pretrain_model'目录为预训练模型目录，'./inference_model'为用于预测的模型目录。
