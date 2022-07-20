# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import collections
import numpy as np
import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Conv2D, BatchNorm, Linear, Dropout, PReLU
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D
from paddle.nn.initializer import XavierNormal, Constant

from .layers import PartialFC
from plsc.models.layers import Model

import math

__all__ = ["IResNet18", "IResNet34", "IResNet50", "IResNet100", "IResNet200"]


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


class FC(nn.Layer):
    def __init__(self,
                 bn_channels,
                 num_channels,
                 num_classes,
                 fc_type,
                 dropout=0.4,
                 name=None,
                 data_format="NCHW"):
        super(FC, self).__init__()
        self.p = dropout
        self.fc_type = fc_type
        self.num_channels = num_channels

        bn_name = "bn_" + name
        if fc_type == "Z":
            self._batch_norm_1 = BatchNorm(
                bn_channels,
                act=None,
                epsilon=1e-05,
                param_attr=ParamAttr(name=bn_name + "_1_scale"),
                bias_attr=ParamAttr(bn_name + "_1_offset"),
                moving_mean_name=bn_name + "_1_mean",
                moving_variance_name=bn_name + "_1_variance",
                data_layout=data_format)
            if self.p > 0:
                self.dropout = Dropout(p=self.p, name=name + '_dropout')

        elif fc_type == "E":
            self._batch_norm_1 = BatchNorm(
                bn_channels,
                act=None,
                epsilon=1e-05,
                param_attr=ParamAttr(name=bn_name + "_1_scale"),
                bias_attr=ParamAttr(bn_name + "_1_offset"),
                moving_mean_name=bn_name + "_1_mean",
                moving_variance_name=bn_name + "_1_variance",
                data_layout=data_format)
            if self.p > 0:
                self.dropout = Dropout(p=self.p, name=name + '_dropout')
            self.fc = Linear(
                num_channels,
                num_classes,
                weight_attr=ParamAttr(
                    initializer=XavierNormal(fan_in=0.0), name=name + ".w_0"),
                bias_attr=ParamAttr(
                    initializer=Constant(), name=name + ".b_0"))
            self._batch_norm_2 = BatchNorm(
                num_classes,
                act=None,
                epsilon=1e-05,
                param_attr=ParamAttr(name=bn_name + "_2_scale"),
                bias_attr=ParamAttr(bn_name + "_2_offset"),
                moving_mean_name=bn_name + "_2_mean",
                moving_variance_name=bn_name + "_2_variance",
                data_layout=data_format)

        elif fc_type == "FC":
            self._batch_norm_1 = BatchNorm(
                bn_channels,
                act=None,
                epsilon=1e-05,
                param_attr=ParamAttr(name=bn_name + "_1_scale"),
                bias_attr=ParamAttr(bn_name + "_1_offset"),
                moving_mean_name=bn_name + "_1_mean",
                moving_variance_name=bn_name + "_1_variance",
                data_layout=data_format)
            self.fc = Linear(
                num_channels,
                num_classes,
                weight_attr=ParamAttr(
                    initializer=XavierNormal(fan_in=0.0), name=name + ".w_0"),
                bias_attr=ParamAttr(
                    initializer=Constant(), name=name + ".b_0"))
            self._batch_norm_2 = BatchNorm(
                num_classes,
                act=None,
                epsilon=1e-05,
                param_attr=ParamAttr(name=bn_name + "_2_scale"),
                bias_attr=ParamAttr(bn_name + "_2_offset"),
                moving_mean_name=bn_name + "_2_mean",
                moving_variance_name=bn_name + "_2_variance",
                data_layout=data_format)

    def forward(self, inputs):
        if self.fc_type == "Z":
            y = self._batch_norm_1(inputs)
            y = paddle.reshape(y, shape=[-1, self.num_channels])
            if self.p > 0:
                y = self.dropout(y)

        elif self.fc_type == "E":
            y = self._batch_norm_1(inputs)
            y = paddle.reshape(y, shape=[-1, self.num_channels])
            if self.p > 0:
                y = self.dropout(y)
            y = self.fc(y)
            y = self._batch_norm_2(y)

        elif self.fc_type == "FC":
            y = self._batch_norm_1(inputs)
            y = paddle.reshape(y, shape=[-1, self.num_channels])
            y = self.fc(y)
            y = self._batch_norm_2(y)

        return y


class IResNet(Model):
    def __init__(self,
                 layers=50,
                 num_features=512,
                 class_num=93431,
                 pfc_config={"model_parallel": True,
                             "sample_ratio": 1.0},
                 fc_type='E',
                 dropout=0.0,
                 input_image_channel=3,
                 input_image_width=112,
                 input_image_height=112,
                 data_format="NCHW"):

        super(IResNet, self).__init__()

        self.layers = layers
        self.data_format = data_format
        self.input_image_channel = input_image_channel

        supported_layers = [18, 34, 50, 100, 200]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(
                supported_layers, layers)

        if layers == 18:
            units = [2, 2, 2, 2]
        elif layers == 34:
            units = [3, 4, 6, 3]
        elif layers == 50:
            units = [3, 4, 14, 3]
        elif layers == 100:
            units = [3, 13, 30, 3]
        elif layers == 200:
            units = [6, 26, 60, 6]

        num_channels = [64, 64, 128, 256]
        num_filters = [64, 128, 256, 512]

        self.conv = ConvBNLayer(
            num_channels=self.input_image_channel,
            num_filters=64,
            filter_size=3,
            stride=1,
            act=None,
            name="conv1",
            data_format=self.data_format)
        self.prelu = PReLU(
            num_parameters=64, data_format=self.data_format, name="prelu1")

        self.block_list = paddle.nn.LayerList()
        for block in range(len(units)):
            shortcut = True
            for i in range(units[block]):
                conv_name = "res" + str(block + 2) + "_" + str(i + 1)
                basic_block = self.add_sublayer(
                    conv_name,
                    BasicBlock(
                        num_channels=num_channels[block]
                        if i == 0 else num_filters[block],
                        num_filters=num_filters[block],
                        stride=2 if shortcut else 1,
                        shortcut=shortcut,
                        name=conv_name,
                        data_format=self.data_format))
                self.block_list.append(basic_block)
                shortcut = False

        assert input_image_width % 16 == 0
        assert input_image_height % 16 == 0
        feat_w = input_image_width // 16
        feat_h = input_image_height // 16
        self.fc_channels = num_filters[-1] * feat_w * feat_h
        #NOTE(GuoxiaWang): don't use NHWC for last fc,
        # thus we can train using NHWC and test using NCHW
        self.fc = FC(num_filters[-1],
                     self.fc_channels,
                     num_features,
                     fc_type,
                     dropout,
                     name='fc',
                     data_format="NCHW")

        pfc_config.update({
            'num_classes': class_num,
            'embedding_size': num_features,
            'name': 'partialfc'
        })
        self.head = PartialFC(**pfc_config)

    def forward(self, inputs):
        if isinstance(inputs, dict):
            x = inputs['data']
        else:
            x = inputs

        x.stop_gradient = True
        if self.data_format == "NHWC":
            x = paddle.tensor.transpose(x, [0, 2, 3, 1])

        y = self.conv(x)
        y = self.prelu(y)
        for block in self.block_list:
            y = block(y)
        if self.data_format == "NHWC":
            y = paddle.tensor.transpose(y, [0, 3, 1, 2])
        y = self.fc(y)

        if not self.training:
            # return embedding feature
            if isinstance(inputs, dict):
                res = {'logits': y}
                if 'targets' in inputs:
                    res['targets'] = inputs['targets']
            else:
                res = y
            return res

        assert isinstance(inputs, dict) and 'targets' in inputs
        y, targets = self.head(y, inputs['targets'])

        return {'logits': y, 'targets': targets}

    def load_pretrained(self, path, rank=0, finetune=False):
        if not os.path.exists(path + '.pdparams'):
            raise ValueError("Model pretrain path {} does not "
                             "exists.".format(path))

        state_dict = paddle.load(path + ".pdparams")

        dist_param_path = path + "_rank{}.pdparams".format(rank)
        if os.path.exists(dist_param_path):
            dist_state_dict = paddle.load(dist_param_path)
            state_dict.update(dist_state_dict)

            # clear
            dist_state_dict.clear()

        if not finetune:
            self.set_dict(state_dict)
            return

        return

    def save(self, path, local_rank=0, rank=0):
        dist_state_dict = collections.OrderedDict()
        state_dict = self.state_dict()
        for name, param in list(state_dict.items()):
            if param.is_distributed:
                dist_state_dict[name] = state_dict.pop(name)

        if local_rank == 0:
            paddle.save(state_dict, path + ".pdparams")

        if len(dist_state_dict) > 0:
            paddle.save(dist_state_dict,
                        path + "_rank{}.pdparams".format(rank))


def IResNet18(**args):
    model = IResNet(layers=18, **args)
    return model


def IResNet34(**args):
    model = IResNet(layers=34, **args)
    return model


def IResNet50(**args):
    model = IResNet(layers=50, **args)
    return model


def IResNet100(**args):
    model = IResNet(layers=100, **args)
    return model


def IResNet200(**args):
    model = IResNet(layers=200, **args)
    return model
