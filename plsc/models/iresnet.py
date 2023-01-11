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
import paddle.nn as nn

from plsc.nn import init

from plsc.nn import PartialFC
from plsc.models.base_model import Model

import math

__all__ = ["IResNet18", "IResNet34", "IResNet50", "IResNet100", "IResNet200"]


def conv3x3(in_planes,
            out_planes,
            stride=1,
            groups=1,
            dilation=1,
            data_format="NCHW"):
    """3x3 convolution with padding"""
    return nn.Conv2D(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        dilation=dilation,
        bias_attr=False,
        data_format=data_format)


def conv1x1(in_planes, out_planes, stride=1, data_format="NCHW"):
    """1x1 convolution"""
    return nn.Conv2D(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias_attr=False,
        data_format=data_format)


class IBasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=\
        1, base_width=64, dilation=1, data_format="NCHW"):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                'Dilation > 1 not supported in BasicBlock')
        self.bn1 = nn.BatchNorm2D(
            inplanes, epsilon=1e-05, data_format=data_format)
        self.conv1 = conv3x3(inplanes, planes, data_format=data_format)
        self.bn2 = nn.BatchNorm2D(
            planes, epsilon=1e-05, data_format=data_format)
        self.prelu = nn.PReLU(planes, data_format=data_format)
        self.conv2 = conv3x3(planes, planes, stride, data_format=data_format)
        self.bn3 = nn.BatchNorm2D(
            planes, epsilon=1e-05, data_format=data_format)
        self.downsample = downsample
        self.stride = stride

    def forward_impl(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out

    def forward(self, x):
        return self.forward_impl(x)


class IResNet(Model):
    def __init__(self,
                 block,
                 layers,
                 dropout=0,
                 num_features=512,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 class_num=93431,
                 pfc_config={"model_parallel": True,
                             "sample_ratio": 1.0},
                 input_image_channel=3,
                 input_image_width=112,
                 input_image_height=112,
                 data_format="NCHW"):
        super(IResNet, self).__init__()

        self.layers = layers
        self.data_format = data_format
        self.input_image_channel = input_image_channel

        assert input_image_width % 16 == 0
        assert input_image_height % 16 == 0
        feat_w = input_image_width // 16
        feat_h = input_image_height // 16
        self.fc_scale = feat_w * feat_h

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                'replace_stride_with_dilation should be None or a 3-element tuple, got {}'
                .format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2D(
            self.input_image_channel,
            self.inplanes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False,
            data_format=data_format)
        self.bn1 = nn.BatchNorm2D(
            self.inplanes, epsilon=1e-05, data_format=data_format)
        self.prelu = nn.PReLU(self.inplanes, data_format=data_format)
        self.layer1 = self._make_layer(
            block, 64, layers[0], stride=2, data_format=data_format)
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            data_format=data_format)
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            data_format=data_format)
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            data_format=data_format)
        self.bn2 = nn.BatchNorm2D(
            512 * block.expansion, epsilon=1e-05, data_format=data_format)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale,
                            num_features)
        self.features = nn.BatchNorm1D(num_features, epsilon=1e-05)
        # self.features = nn.BatchNorm1D(num_features, epsilon=1e-05, weight_attr=False)

        for m in self.sublayers():
            if isinstance(m, paddle.nn.Conv2D):
                init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (paddle.nn.BatchNorm2D, paddle.nn.GroupNorm)):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.sublayers():
                if isinstance(m, IBasicBlock):
                    init.constant_(m.bn2.weight, 0)

        pfc_config.update({
            'num_classes': class_num,
            'embedding_size': num_features,
            'name': 'partialfc'
        })
        self.head = PartialFC(**pfc_config)

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    stride=1,
                    dilate=False,
                    data_format="NCHW"):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(
                    self.inplanes,
                    planes * block.expansion,
                    stride,
                    data_format=data_format),
                nn.BatchNorm2D(
                    planes * block.expansion,
                    epsilon=1e-05,
                    data_format=data_format))
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                data_format=data_format))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    data_format=data_format))
        return nn.Sequential(*layers)

    def forward(self, inputs):

        if self.training:
            with paddle.no_grad():
                # Note(GuoxiaWang)
                # self.features = nn.BatchNorm1D(num_features, epsilon=1e-05, weight_attr=False)
                self.features.weight.fill_(1.0)

        if isinstance(inputs, dict):
            x = inputs['data']
        else:
            x = inputs

        x.stop_gradient = True
        if self.data_format == "NHWC":
            x = paddle.tensor.transpose(x, [0, 2, 3, 1])

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)

        if self.data_format == "NHWC":
            x = paddle.tensor.transpose(x, [0, 3, 1, 2])

        x = paddle.flatten(x, 1)
        x = self.dropout(x)

        x = self.fc(x)
        x = self.features(x)

        if not self.training:
            # return embedding feature
            if isinstance(inputs, dict):
                res = {'logits': x}
                if 'targets' in inputs:
                    res['targets'] = inputs['targets']
            else:
                res = x
            return res

        assert isinstance(inputs, dict) and 'targets' in inputs
        x, targets = self.head(x, inputs['targets'])

        return {'logits': x, 'targets': targets}

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


def IResNet18(**kwargs):
    model = IResNet(IBasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def IResNet34(**kwargs):
    model = IResNet(IBasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def IResNet50(**kwargs):
    model = IResNet(IBasicBlock, [3, 4, 14, 3], **kwargs)
    return model


def IResNet100(**kwargs):
    model = IResNet(IBasicBlock, [3, 13, 30, 3], **kwargs)
    return model


def IResNet200(**kwargs):
    model = IResNet(IBasicBlock, [6, 26, 60, 6], **kwargs)
    return model
