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
from paddle import distributed as dist
from paddle.fluid.framework import EagerParamBase

from plsc.utils import logger


def _all_gather(tensor, group=None):
    tensor_shape = list(tensor.shape)
    tensor_shape[0] *= group.nranks
    out = paddle.empty(tensor_shape, tensor.dtype)
    out.stop_gradient = tensor.stop_gradient
    task = group.process_group.all_gather(tensor, out)
    task.wait()
    return out


class AllGather(paddle.autograd.PyLayer):
    """AllGather op with gradient backward"""

    @staticmethod
    def forward(ctx, tensor, group=None):
        ctx.group = group
        out = _all_gather(tensor, group)
        return out

    @staticmethod
    def backward(ctx, grad):
        group = ctx.group
        grad_list = paddle.split(grad, group.nranks, axis=0)
        rank = group.get_group_rank(dist.get_rank())
        grad_out = grad_list[rank]

        dist_ops = [
            group.process_group.reduce(grad_out, rank,
                                       paddle.fluid.core.ReduceOp.SUM)
            if i == rank else group.process_group.reduce(
                grad_list[i], i, paddle.fluid.core.ReduceOp.SUM)
            for i in range(group.nranks)
        ]
        for _op in dist_ops:
            _op.wait()

        grad_out *= len(grad_list)  # cooperate with distributed loss function
        return grad_out


def all_gather(tensor, axis=0, group=None):

    group = dist.collective._get_default_group() if group is None else group

    if not tensor.stop_gradient:
        output = AllGather.apply(tensor, group=group)
    else:
        output = _all_gather(tensor, group)

    if axis != 0:
        output = paddle.concat(
            paddle.split(
                output, group.nranks, axis=0), axis=axis)
    return output


class PartialFC(nn.Layer):
    """
    Author: {Xiang An, Yang Xiao, XuHan Zhu} in DeepGlint,
    Partial FC: Training 10 Million Identities on a Single Machine
    See the original paper:
    https://arxiv.org/abs/2010.05222
    """

    @paddle.no_grad()
    def __init__(self,
                 num_classes,
                 embedding_size=512,
                 sample_ratio=1.0,
                 model_parallel=False,
                 sparse_update=True,
                 name=None):
        super(PartialFC, self).__init__()
        self.num_classes: int = num_classes
        self.sample_ratio: float = sample_ratio
        self.embedding_size: int = embedding_size
        self.model_parallel: bool = model_parallel
        self.sparse_update: bool = sparse_update

        assert self.sample_ratio > 0 and self.sample_ratio <= 1.0

        rank = 0
        world_size = 1
        if self.model_parallel:
            assert paddle.fluid.core.is_compiled_with_dist()
            rank = paddle.distributed.get_rank()
            world_size = paddle.distributed.get_world_size()

            if world_size == 1:
                logger.warning("model_parallel is modified to False."
                               " world_size must greater than"
                               " 1 when model_parallel=True.")
                self.model_parallel = False

        # Default we use model parallel when group=None.
        # When group=False, it is equal to data parallel.
        self.group = None
        if not self.model_parallel:
            self.group = False

        self.rank = rank
        self.world_size = world_size

        self.num_local: int = num_classes // self.world_size + int(
            self.rank < num_classes % self.world_size)
        self.class_start: int = num_classes // self.world_size * self.rank + min(
            self.rank, num_classes % self.world_size)
        self.num_sample: int = int(self.sample_ratio * self.num_local)

        if model_parallel and world_size > 0:
            if name is None:
                name = 'dist@partialfc@rank@%05d.w' % rank
            else:
                name = name + '@dist@rank@%05d.w' % rank
        else:
            if name is None:
                name = 'partialfc.w'

        param_attr = paddle.ParamAttr(
            name=name,
            initializer=paddle.nn.initializer.Normal(
                mean=0, std=0.01))

        self.index = None
        self.weight = self.create_parameter(
            shape=[self.num_local, self.embedding_size],
            attr=param_attr,
            is_bias=False)
        self.weight.is_distributed = self.model_parallel

        # NOTE(GuoxiaWang): stop full gradient and set has_sparse_grad attr,
        # has_sparse_grad be used to sparse_momentum
        if self.sample_ratio < 1.0 and self.model_parallel and self.sparse_update:
            setattr(self.weight, 'has_sparse_grad', True)
            self.weight.stop_gradient = True
        self.sub_weight = None

    @paddle.no_grad()
    def class_center_sample(self, labels):

        labels = labels.reshape((-1, 1))
        index_positive = (self.class_start <= labels) & (
            labels < self.class_start + self.num_local)

        local_label = labels[index_positive] - self.class_start

        positive = paddle.unique(local_label)
        if self.num_sample - positive.shape[0] >= 0:
            perm = paddle.rand([self.num_local])
            perm[positive] = 2.0
            index = paddle.topk(perm, k=self.num_sample)[1]
            index = paddle.sort(index)
        else:
            index = positive

        local_sampled_ids = index + self.class_start
        sampled_ids = all_gather(local_sampled_ids, axis=0)

        labels = paddle.searchsorted(sampled_ids, labels)

        return labels, index

    def forward(self, feature, label):
        if self.model_parallel:
            total_feature = all_gather(feature, axis=0)

            label.stop_gradient = True
            total_label = all_gather(label, axis=0)
        else:
            total_feature = feature
            total_label = label

        if self.sample_ratio < 1.0:
            # partial fc sample process
            total_label, self.index = self.class_center_sample(total_label)
            total_label.stop_gradient = True
            self.index.stop_gradient = True
            self.sub_weight = paddle.gather(self.weight, self.index, axis=0)

            # NOTE(GuoxiaWang): stop generate the full gradient 
            # when use partial fc in model parallel,
            # but it requires sub gradient.
            if self.model_parallel and self.sparse_update:
                self.sub_weight.stop_gradient = False

                def sparse_grad_hook_fn():
                    setattr(self.weight, 'index', self.index)
                    setattr(self.weight, 'axis', 0)
                    self.weight._set_grad_ivar(self.sub_weight.grad)

                self.sub_weight._register_backward_hook(sparse_grad_hook_fn)

        else:
            self.sub_weight = self.weight

        norm_feature = paddle.fluid.layers.l2_normalize(total_feature, axis=1)
        norm_weight = paddle.fluid.layers.l2_normalize(self.sub_weight, axis=1)

        local_logit = paddle.matmul(
            norm_feature, norm_weight, transpose_y=True)
        return local_logit, total_label
