# Quick Start
## How to install PLSC

PLSC requires python 2.7 or above versions.

First, you have to install PaddlePaddle. Using the following command to install paddlepaddle:
```shell script
pip install paddlepaddle-gpu>=1.6.2
```
For now, paddlepaddle-gpu is required.

Then, install PLSC using pip:

```shell script
pip install plsc
```

## How to do train and validation
The following example code shows how to train a model with PLSC.
```python
from plsc import Entry

if __name__ == "__main__":
    ins = Entry()
    ins.set_train_epochs(1)
    ins.set_model_save_dir("./saved_model")
    # ins.set_with_test(False)  # 当没有验证集时，请取消该行的注释
    # ins.set_loss_type('arcface')  # 当仅有一张GPU卡时，请取消该行的注释
    ins.train()
```

And use the following command to start the train.
```shell script
python -m paddle.distributed.launch \
    --cluster_node_ips="127.0.0.1" \
    --node_ip="127.0.0.1" \
    --selected_gpus=0,1,2,3,4,5,6,7 \
    train.py
```

The following example code shows how to do model validation with PLSC.
```python
from plsc import Entry

if __name__ == "__main__":
    ins = Entry()
    ins.set_checkpoint_dir("./saved_model/0/")
    ins.test()
```
