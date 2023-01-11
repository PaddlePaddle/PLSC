# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
'''
Adapted from https://github.com/cavalleria/cavaface.pytorch/blob/master/backbone/mobilefacenet.py
Original author cavalleria
'''
import os
import collections

import paddle
from paddle import nn
import math

from plsc.nn import init

from plsc.nn import PartialFC
from plsc.models.base_model import Model

__all__ = ['MobileFaceNet_base', 'MobileFaceNet_large', 'MobileFaceNet']


class Flatten(nn.Layer):
    def forward(self, x):
        return paddle.flatten(x, 1)


class ConvBlock(nn.Layer):
    def __init__(self,
                 in_c,
                 out_c,
                 kernel=(1, 1),
                 stride=(1, 1),
                 padding=(0, 0),
                 groups=1,
                 data_format="NCHW"):
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2D(
                in_c,
                out_c,
                kernel_size=kernel,
                groups=groups,
                stride=stride,
                padding=padding,
                bias_attr=False,
                data_format=data_format),
            nn.BatchNorm2D(
                out_c, epsilon=1e-05, data_format=data_format),
            nn.PReLU(
                out_c, data_format=data_format))

    def forward(self, x):
        return self.layers(x)


class LinearBlock(nn.Layer):
    def __init__(self,
                 in_c,
                 out_c,
                 kernel=(1, 1),
                 stride=(1, 1),
                 padding=(0, 0),
                 groups=1,
                 data_format="NCHW"):
        super(LinearBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2D(
                in_c,
                out_c,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                groups=groups,
                bias_attr=False,
                data_format=data_format),
            nn.BatchNorm2D(
                out_c, epsilon=1e-05, data_format=data_format), )

    def forward(self, x):
        return self.layers(x)


class DepthWise(nn.Layer):
    def __init__(self,
                 in_c,
                 out_c,
                 residual=False,
                 kernel=(3, 3),
                 stride=(2, 2),
                 padding=(1, 1),
                 groups=1,
                 data_format="NCHW"):
        super(DepthWise, self).__init__()
        self.residual = residual
        self.layers = nn.Sequential(
            ConvBlock(
                in_c,
                out_c=groups,
                kernel=(1, 1),
                padding=(0, 0),
                stride=(1, 1),
                data_format=data_format),
            ConvBlock(
                groups,
                groups,
                groups=groups,
                kernel=kernel,
                padding=padding,
                stride=stride,
                data_format=data_format),
            LinearBlock(
                groups,
                out_c,
                kernel=(1, 1),
                padding=(0, 0),
                stride=(1, 1),
                data_format=data_format))

    def forward(self, x):
        short_cut = None
        if self.residual:
            short_cut = x
        x = self.layers(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class Residual(nn.Layer):
    def __init__(self,
                 c,
                 num_block,
                 groups,
                 kernel=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1),
                 data_format="NCHW"):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(
                DepthWise(
                    c,
                    c,
                    True,
                    kernel,
                    stride,
                    padding,
                    groups,
                    data_format=data_format))
        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.layers(x)


class GDC(nn.Layer):
    def __init__(self, embedding_size):
        super(GDC, self).__init__()
        self.layers = nn.Sequential(
            LinearBlock(
                512,
                512,
                groups=512,
                kernel=(7, 7),
                stride=(1, 1),
                padding=(0, 0)),
            Flatten(),
            nn.Linear(
                512, embedding_size, bias_attr=False),
            nn.BatchNorm1D(
                embedding_size, epsilon=1e-05))

    def forward(self, x):
        return self.layers(x)


class MobileFaceNet(Model):
    def __init__(self,
                 num_features=512,
                 blocks=(1, 4, 6, 2),
                 scale=2,
                 class_num=93431,
                 pfc_config={"model_parallel": True,
                             "sample_ratio": 1.0},
                 data_format="NCHW",
                 **args):
        super().__init__()
        self.data_format = data_format
        self.scale = scale

        self.layers = nn.LayerList()
        self.layers.append(
            ConvBlock(
                3,
                64 * self.scale,
                kernel=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                data_format=data_format))
        if blocks[0] == 1:
            self.layers.append(
                ConvBlock(
                    64 * self.scale,
                    64 * self.scale,
                    kernel=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                    groups=64,
                    data_format=data_format))
        else:
            self.layers.append(
                Residual(
                    64 * self.scale,
                    num_block=blocks[0],
                    groups=128,
                    kernel=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1)),
                data_format=data_format)

        self.layers.extend([
            DepthWise(
                64 * self.scale,
                64 * self.scale,
                kernel=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                groups=128,
                data_format=data_format),
            Residual(
                64 * self.scale,
                num_block=blocks[1],
                groups=128,
                kernel=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                data_format=data_format),
            DepthWise(
                64 * self.scale,
                128 * self.scale,
                kernel=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                groups=256,
                data_format=data_format),
            Residual(
                128 * self.scale,
                num_block=blocks[2],
                groups=256,
                kernel=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                data_format=data_format),
            DepthWise(
                128 * self.scale,
                128 * self.scale,
                kernel=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                groups=512,
                data_format=data_format),
            Residual(
                128 * self.scale,
                num_block=blocks[3],
                groups=256,
                kernel=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                data_format=data_format),
        ])

        self.conv_sep = ConvBlock(
            128 * self.scale,
            512,
            kernel=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            data_format=data_format)
        self.features = GDC(num_features)
        self._initialize_weights()

        pfc_config.update({
            'num_classes': class_num,
            'embedding_size': num_features,
            'name': 'partialfc'
        })
        self.head = PartialFC(**pfc_config)

    def _initialize_weights(self):
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2D):
                init.ones_(m.weight)
                init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, inputs):
        if isinstance(inputs, dict):
            x = inputs['data']
        else:
            x = inputs

        x.stop_gradient = True
        if self.data_format == "NHWC":
            x = paddle.tensor.transpose(x, [0, 2, 3, 1])

        for func in self.layers:
            x = func(x)
        x = self.conv_sep(x)

        if self.data_format == "NHWC":
            x = paddle.tensor.transpose(x, [0, 3, 1, 2])

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


def MobileFaceNet_base(**args):
    model = MobileFaceNet(blocks=(1, 4, 6, 2), scale=2, **args)
    return model


def MobileFaceNet_large(**args):
    model = MobileFaceNet(blocks=(2, 8, 12, 4), scale=4, **args)
    return model
