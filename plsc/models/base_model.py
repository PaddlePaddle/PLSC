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

import math

import paddle.fluid as fluid
from paddle.fluid import unique_name

from . import dist_algo

__all__ = ["BaseModel"]


class BaseModel(object):
    """
    Base class for custom models.
    The sub-class must implement the build_network method,
    which constructs the custom model. And we will add the
    distributed fc layer for you automatically.
    """

    def __init__(self):
        super(BaseModel, self).__init__()

    def build_network(self, input, label, is_train=True):
        """
        Construct the custom model, and we will add the distributed fc layer
        at the end of your model automatically.
        """
        raise NotImplementedError(
            "You must implement this method in your subclass.")

    def get_output(self,
                   input,
                   label,
                   num_classes,
                   num_ranks=1,
                   rank_id=0,
                   is_train=True,
                   param_attr=None,
                   bias_attr=None,
                   loss_type="dist_softmax",
                   margin=0.5,
                   scale=64.0):
        """
        Add the distributed fc layer for the custom model.

        Params:
            input: input for the model
            label: label for the input
            num_classes: number of classes for the classifier
            num_ranks: number of trainers, i.e., GPUs
            rank_id: id for the current trainer, from 0 to num_ranks - 1
            is_train: build the network for training or not
            param_attr: param_attr for the weight parameter of fc
            bias_attr: bias_attr for the weight parameter for fc
            loss_type: loss type to use, one of dist_softmax, softmax, arcface
                and dist_arcface
            margin: the margin parameter for arcface and dist_arcface
            scale: the scale parameter for arcface and dist_arcface
        """
        supported_loss_types = ["dist_softmax", "dist_arcface",
                                "softmax", "arcface"]
        assert loss_type in supported_loss_types, \
            "Supported loss types: {}, but given: {}".format(
                supported_loss_types, loss_type)

        emb = self.build_network(input, label, is_train)
        prob = None
        loss = None
        if loss_type == "softmax":
            loss, prob = BaseModel._fc_classify(emb,
                                                label,
                                                num_classes,
                                                param_attr,
                                                bias_attr)
        elif loss_type == "arcface":
            loss, prob = BaseModel._arcface(emb,
                                            label,
                                            num_classes,
                                            param_attr,
                                            margin,
                                            scale)
        elif loss_type == "dist_arcface":
            loss = dist_algo.distributed_arcface_classify(x=emb,
                                                          label=label,
                                                          class_num=num_classes,
                                                          nranks=num_ranks,
                                                          rank_id=rank_id,
                                                          margin=margin,
                                                          logit_scale=scale,
                                                          param_attr=param_attr)
        elif loss_type == "dist_softmax":
            loss = dist_algo.distributed_softmax_classify(x=emb,
                                                          label=label,
                                                          class_num=num_classes,
                                                          nranks=num_ranks,
                                                          rank_id=rank_id,
                                                          param_attr=param_attr,
                                                          use_bias=True,
                                                          bias_attr=bias_attr)

        return emb, loss, prob

    @staticmethod
    def _fc_classify(input, label, out_dim, param_attr, bias_attr):
        if param_attr is None:
            stddev = 1.0 / math.sqrt(input.shape[1] * 1.0)
            param_attr = fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stddev, stddev))

        out = fluid.layers.fc(input=input,
                              size=out_dim,
                              param_attr=param_attr,
                              bias_attr=bias_attr)
        loss, prob = fluid.layers.softmax_with_cross_entropy(
            logits=out,
            label=label,
            return_softmax=True)
        avg_loss = fluid.layers.mean(x=loss)
        return avg_loss, prob

    @staticmethod
    def _arcface(input, label, out_dim, param_attr, margin, scale):
        input_norm = fluid.layers.sqrt(
            fluid.layers.reduce_sum(fluid.layers.square(input), dim=1))
        input = fluid.layers.elementwise_div(input, input_norm, axis=0)

        if param_attr is None:
            param_attr = fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Xavier(uniform=False, fan_in=0.0))
        weight = fluid.layers.create_parameter(
            shape=[input.shape[1], out_dim],
            dtype='float32',
            name=unique_name.generate('final_fc_w'),
            attr=param_attr)

        weight_norm = fluid.layers.sqrt(
            fluid.layers.reduce_sum(fluid.layers.square(weight), dim=0))
        weight = fluid.layers.elementwise_div(weight, weight_norm, axis=1)
        cos = fluid.layers.mul(input, weight)

        theta = fluid.layers.acos(cos)
        margin_cos = fluid.layers.cos(theta + margin)
        one_hot = fluid.layers.one_hot(label, out_dim)
        diff = (margin_cos - cos) * one_hot
        target_cos = cos + diff
        logit = fluid.layers.scale(target_cos, scale=scale)

        loss, prob = fluid.layers.softmax_with_cross_entropy(
            logits=logit,
            label=label,
            return_softmax=True)
        avg_loss = fluid.layers.mean(x=loss)

        one_hot.stop_gradient = True

        return avg_loss, prob
