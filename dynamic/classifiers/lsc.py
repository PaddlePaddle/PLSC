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

import math
import os
import paddle
import paddle.nn as nn


class LargeScaleClassifier(nn.Layer):
    """
    Author: {Xiang An, Yang Xiao, XuHan Zhu} in DeepGlint,
    Partial FC: Training 10 Million Identities on a Single Machine
    See the original paper:
    https://arxiv.org/abs/2010.05222
    """

    @paddle.no_grad()
    def __init__(self,
                 rank,
                 world_size,
                 num_classes,
                 margin1=1.0,
                 margin2=0.5,
                 margin3=0.0,
                 scale=64.0,
                 sample_ratio=1.0,
                 embedding_size=512,
                 fp16=False,
                 name=None):
        super(LargeScaleClassifier, self).__init__()
        self.num_classes: int = num_classes
        self.rank: int = rank
        self.world_size: int = world_size
        self.sample_ratio: float = sample_ratio
        self.embedding_size: int = embedding_size
        self.fp16 = fp16
        self.num_local: int = (num_classes + world_size - 1) // world_size
        if num_classes % world_size != 0 and rank == world_size - 1:
            self.num_local = num_classes % self.num_local
        self.num_sample: int = int(self.sample_ratio * self.num_local)
        self.margin1 = margin1
        self.margin2 = margin2
        self.margin3 = margin3
        self.logit_scale = scale

        self._parameter_list = []

        if name is None:
            name = 'dist@fc@rank@%05d.w' % rank

        stddev = math.sqrt(2.0 / (self.embedding_size + self.num_local))
        param_attr = paddle.ParamAttr(
            name=name, initializer=paddle.nn.initializer.Normal(std=stddev))

        self.index = None
        self.weight = self.create_parameter(
            shape=[self.embedding_size, self.num_local],
            attr=param_attr,
            is_bias=False,
            dtype='float16' if self.fp16 else 'float32')

        if int(self.sample_ratio) < 1:
            self.weight.stop_gradient = True

    def step(self, optimizer):
        if int(self.sample_ratio) < 1:
            velocity = optimizer._accumulators['velocity'][self.weight.name]
            _, _ = paddle._C_ops.sparse_momentum(
                self.weight,
                self._parameter_list[0].grad,
                velocity,
                self.index,
                paddle.to_tensor(
                    optimizer.get_lr(), dtype='float32'),
                self.weight,
                velocity,
                'mu',
                optimizer._momentum,
                'use_nesterov',
                optimizer._use_nesterov,
                'regularization_method',
                optimizer._regularization_method,
                'regularization_coeff',
                optimizer._regularization_coeff,
                'axis',
                1)

    def clear_grad(self):
        self._parameter_list = []

    def forward(self, feature, label):
        if self.world_size > 1:
            feature_list = []
            paddle.distributed.all_gather(feature_list, feature)
            total_feature = paddle.concat(feature_list, axis=0)

            label_list = []
            paddle.distributed.all_gather(label_list, label)
            total_label = paddle.concat(label_list, axis=0)
            total_label.stop_gradient = True
        else:
            total_feature = feature
            total_label = label

        if self.sample_ratio < 1.0:
            # partial fc sample process
            total_label, self.index = paddle.nn.functional.class_center_sample(
                total_label, self.num_local, self.num_sample)
            total_label.stop_gradient = True
            self.index.stop_gradient = True
            self.sub_weight = paddle.gather(self.weight, self.index, axis=1)
            self.sub_weight.stop_gradient = False
            self._parameter_list.append(self.sub_weight)
            if self.sub_weight.dtype == paddle.float16:
                self.sub_weight = paddle.cast(self.sub_weight, dtype='float32')
        else:
            self.sub_weight = self.weight

        if total_feature.dtype == paddle.float16:
            total_feature = paddle.cast(total_feature, dtype='float32')

        norm_feature = paddle.nn.functional.normalize(total_feature, axis=1)
        norm_weight = paddle.nn.functional.normalize(self.sub_weight, axis=0)
        local_logit = paddle.matmul(norm_feature, norm_weight)

        loss = paddle.nn.functional.margin_cross_entropy(
            local_logit,
            total_label,
            margin1=self.margin1,
            margin2=self.margin2,
            margin3=self.margin3,
            scale=self.logit_scale,
            return_softmax=False)

        return loss
