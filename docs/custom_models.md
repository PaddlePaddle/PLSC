# 自定义模型

默认地，PaddlePaddle大规模分类库构建基于ResNet50模型的训练模型。

PLSC提供了模型基类plsc.models.base_model.BaseModel，用户可以基于该基类构建自己的网络模型。用户自定义的模型类需要继承自该基类，并实现build_network方法，该方法用于构建用户自定义模型。

下面的例子给出如何使用BaseModel基类定义用户自己的网络模型, 以及如何使用。
```python
import paddle.fluid as fluid
import plsc.entry as entry
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

        conv = self.conv_bn_layer(
            input=input, num_filters=64, filter_size=3, stride=1,
            pad=1, act='prelu', is_train=is_train)

        for block in range(len(depth)):
            for i in range(depth[block]):
                conv = self.bottleneck_block(
                    input=conv,
                    num_filters=num_filters[block],
                    stride=2 if i == 0 else 1,
                    is_train=is_train)

        bn = fluid.layers.batch_norm(input=conv, act=None, epsilon=2e-05,
            is_test=False if is_train else True)
        drop = fluid.layers.dropout(x=bn, dropout_prob=0.4,
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
        emb = fluid.layers.batch_norm(input=fc, act=None, epsilon=2e-05,
            is_test=False if is_train else True)
        return emb

        ... ...

if __name__ == "__main__":
    ins = entry.Entry()
    ins.set_model(ResNet())
    ins.train()
```

用户自定义模型类需要继承自基类BaseModel，并实现build_network方法，实现用户的自定义模型。

build_network方法的输入如下：
* input: 输入图像数据
* label: 图像类别
* is_train: 表示训练阶段还是测试/预测阶段

build_network方法返回用户自定义组网的输出变量，BaseModel类的get_output方法将调用该方法获取用户自定义组网的输出，并自动在其后添加分布式FC层。

