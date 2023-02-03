# copyright (c) 2023 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Multi-task heads.
Only defined convolution blocks.
"""
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn import Conv2D, BatchNorm, Linear, \
    MaxPool2D, Dropout, PReLU


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None,
                 name=None,
                 data_format="NCHW"):
        super(ConvBNLayer, self).__init__()

        self._conv = Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False,
            data_format=data_format)
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        self._batch_norm = BatchNorm(
            num_filters,
            act=act,
            epsilon=1e-05,
            param_attr=ParamAttr(name=bn_name + "_scale"),
            bias_attr=ParamAttr(bn_name + "_offset"),
            moving_mean_name=bn_name + "_mean",
            moving_variance_name=bn_name + "_variance",
            data_layout=data_format)

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y


class ConvBNLayerAttr(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 padding=0,
                 act=None,
                 name=None):
        super(ConvBNLayerAttr, self).__init__()

        self._conv = Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            weight_attr=ParamAttr(),
            bias_attr=False)
        self._batch_norm = BatchNorm(num_filters, act=act)

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y


class BasicBlock(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 shortcut=True,
                 name=None,
                 data_format="NCHW"):
        super(BasicBlock, self).__init__()
        self.stride = stride
        bn_name = "bn_" + name[3:] + "_before"
        self._batch_norm = BatchNorm(
            num_channels,
            act=None,
            epsilon=1e-05,
            param_attr=ParamAttr(name=bn_name + "_scale"),
            bias_attr=ParamAttr(bn_name + "_offset"),
            moving_mean_name=bn_name + "_mean",
            moving_variance_name=bn_name + "_variance",
            data_layout=data_format)

        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=3,
            stride=1,
            act=None,
            name=name + "_branch2a",
            data_format=data_format)
        self.prelu = PReLU(
            num_parameters=num_filters,
            data_format=data_format,
            name=name + "_branch2a_prelu")
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act=None,
            name=name + "_branch2b",
            data_format=data_format)

        if shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters,
                filter_size=1,
                stride=stride,
                act=None,
                name=name + "_branch1",
                data_format=data_format)

        self.shortcut = shortcut

    def forward(self, inputs):
        y = self._batch_norm(inputs)
        y = self.conv0(y)
        y = self.prelu(y)
        conv1 = self.conv1(y)

        if self.shortcut:
            short = self.short(inputs)
        else:
            short = inputs
        y = paddle.add(x=short, y=conv1)
        return y


class TaskBlock(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride=1,
                 shortcut=True,
                 padding=0,
                 name=None,
                 class_num=10,
                 task_occlu=False,
                 data_format="NCHW"):
        super(TaskBlock, self).__init__()
        self.conv_for_fc = ConvBNLayerAttr(
            num_channels=num_channels,
            num_filters=64,
            filter_size=3,
            stride=1,
            padding=1,
            act=None,
            name="conv_for_fc")
        self.prelu_bottom = PReLU(
            num_parameters=64, data_format=data_format, name="prelu_bottom")

        self.conv0 = ConvBNLayerAttr(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=1,
            padding=padding,
            act=None)
        self.pool = MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.fc0 = Linear(256, 128)
        self.prelu0 = PReLU(num_parameters=64)
        self.prelu1 = PReLU(num_parameters=128)
        self.task_occlu = task_occlu
        self.fc1 = Linear(128, class_num)

    def forward(self, inputs):
        y = self.conv_for_fc(inputs)
        # y = self.prelu_bottom(y)
        y = self.conv0(y)
        y = self.prelu0(y)
        N = y.shape[0]
        y = self.pool(y)
        y = paddle.reshape(y, [N, -1])  # 128, 64 * 3 * 3
        y = self.fc0(y)
        y = self.prelu1(y)
        out = self.fc1(y)
        return out
