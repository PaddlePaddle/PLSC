# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.fluid as fluid
import math
import os
import numpy as np
from paddle.fluid import unique_name
from .base_model import BaseModel


__all__ = ["ResNet", "ResNet50", "ResNet101", "ResNet152"]


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

    def conv_bn_layer(self,
                      input,
                      num_filters,
                      filter_size,
                      stride=1,
                      pad=0,
                      groups=1,
                      is_train=True,
                      act=None):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=pad,
            groups=groups,
            act=None,
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Xavier(
                    uniform=False, fan_in=0.0)),
            bias_attr=False)
        if act == 'prelu':
            bn = fluid.layers.batch_norm(input=conv, act=None, epsilon=2e-05,
                momentum=0.9, is_test=False if is_train else True)
            return fluid.layers.prelu(bn, mode="all",
                param_attr=fluid.param_attr.ParamAttr(
                    initializer=fluid.initializer.Constant(0.25)))
        else:
            return fluid.layers.batch_norm(input=conv, act=act, epsilon=2e-05,
                is_test=False if is_train else True)

    def shortcut(self, input, ch_out, stride, is_train):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1:
            return self.conv_bn_layer(input, ch_out, 1, stride,
                is_train=is_train)
        else:
            return input

    def bottleneck_block(self, input, num_filters, stride, is_train):
        if self.layers < 101:
            bn1 = fluid.layers.batch_norm(input=input, act=None, epsilon=2e-05,
                is_test=False if is_train else True)
            conv1 = self.conv_bn_layer(
                input=bn1, num_filters=num_filters, filter_size=3, pad=1,
                act='prelu', is_train=is_train)
            conv2 = self.conv_bn_layer(
                input=conv1, num_filters=num_filters, filter_size=3,
                stride=stride, pad=1, act=None, is_train=is_train)
        else:
            bn0 = fluid.layers.batch_norm(input=input, act=None, epsilon=2e-05,
                is_test=False if is_train else True)
            conv0 = self.conv_bn_layer(
                input=bn0, num_filters=num_filters/4, filter_size=1, pad=0,
                act='prelu', is_train=is_train)
            conv1 = self.conv_bn_layer(
                input=conv0, num_filters=num_filters/4, filter_size=3, pad=1,
                act='prelu', is_train=is_train)
            conv2 = self.conv_bn_layer(
                input=conv1, num_filters=num_filters, filter_size=1,
                stride=stride, pad=0, act=None, is_train=is_train)

        short = self.shortcut(input, num_filters, stride, is_train=is_train)
        return fluid.layers.elementwise_add(x=short, y=conv2, act=None)


def ResNet50(emb_dim=512):
    model = ResNet(layers=50, emb_dim=emb_dim)
    return model


def ResNet101(emb_dim=512):
    model = ResNet(layers=101, emb_dim=emb_dim)
    return model


def ResNet152(emb_dim=512):
    model = ResNet(layers=152, emb_dim=emb_dim)
    return model
