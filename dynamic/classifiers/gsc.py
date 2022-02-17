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
import numpy as np
import os
import paddle
import paddle.nn as nn


class FakeDDPGroup(paddle.distributed.collective.Group):
    def __init__(self):
        self.rank = 0
        self.nranks = 1
        self.id = 0
        self.ranks = [0]

    def is_member(self):
        return True

    def get_group_rank(self, rank):
        return 0


class GeneralScaleClassifier(nn.Layer):
    """
    DistributedDataParallel PartialFC

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
                 numpy_init=True,
                 name=None):
        super(GeneralScaleClassifier, self).__init__()
        self.num_classes: int = num_classes
        self.rank: int = rank
        self.world_size: int = world_size
        self.sample_ratio: float = sample_ratio
        self.embedding_size: int = embedding_size
        self.fp16 = fp16
        self.num_sample: int = int(self.sample_ratio * self.num_classes)
        self.margin1 = margin1
        self.margin2 = margin2
        self.margin3 = margin3
        self.logit_scale = scale

        # construct a fake DistributedDataParallel group
        self.group = FakeDDPGroup()

        if name is None:
            name = 'gfc.w'

        if numpy_init:
            stddev = math.sqrt(2.0 / (self.embedding_size + self.num_classes))
            init_total = np.random.RandomState(2021).normal(0.0, stddev, (
                self.embedding_size, self.num_classes))
            param_attr = paddle.ParamAttr(
                name=name,
                initializer=paddle.fluid.initializer.NumpyArrayInitializer(
                    init_total))
        else:
            stddev = math.sqrt(2.0 / (self.embedding_size + self.num_classes))
            param_attr = paddle.ParamAttr(
                name=name,
                initializer=paddle.nn.initializer.Normal(std=stddev))

        self.index = None
        self.weight = self.create_parameter(
            shape=[self.embedding_size, self.num_classes],
            attr=param_attr,
            is_bias=False,
            dtype='float16' if self.fp16 else 'float32')

        # NOTE(GuoxiaWang): stop full gradient and set is_sparse_grad attr
        # is_sparse_grad be used fo sparse_momentum
        if self.sample_ratio < 1.0:
            setattr(self.weight, 'is_sparse_grad', True)
            self.weight.stop_gradient = True
        self.sub_weight = None

    def set_attr_for_sparse_momentum(self):
        # The attribute are used for sparse_momentum
        if getattr(self.weight, 'is_sparse_grad', None):
            setattr(self.weight, 'sparse_grad', self.sub_weight.grad)
            setattr(self.weight, 'index', self.index)
            setattr(self.weight, 'axis', 1)

    def forward(self, feature, label):

        total_feature = feature
        total_label = label

        if self.sample_ratio < 1.0:
            # partial fc sample process
            total_label, self.index = paddle.nn.functional.class_center_sample(
                total_label,
                self.num_classes,
                self.num_sample,
                group=self.group)
            total_label.stop_gradient = True
            self.index.stop_gradient = True
            self.sub_weight = paddle.gather(self.weight, self.index, axis=1)

            # NOTE(GuoxiaWang): stop generate the full gradient when use partial fc,
            # but it need sub gradient.
            self.sub_weight.stop_gradient = False

        else:
            self.sub_weight = self.weight

        norm_feature = paddle.fluid.layers.l2_normalize(total_feature, axis=1)
        norm_weight = paddle.fluid.layers.l2_normalize(self.sub_weight, axis=0)

        local_logit = paddle.matmul(norm_feature, norm_weight)

        loss = paddle.nn.functional.margin_cross_entropy(
            local_logit,
            total_label,
            margin1=self.margin1,
            margin2=self.margin2,
            margin3=self.margin3,
            scale=self.logit_scale,
            return_softmax=False,
            reduction=None,
            group=self.group, )

        loss = paddle.mean(loss)

        return loss
