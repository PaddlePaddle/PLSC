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

from __future__ import division
from __future__ import print_function

import math

import logging
import paddle
import paddle.nn as nn
import paddle.utils.unique_name as unique_name
from paddle.optimizer import Optimizer
from paddle.distributed.fleet.utils import class_center_sample
from ..utils.fp16_utils import rewrite_program, update_role_var_grad
from ..utils.fp16_utils import update_loss_scaling, move_optimize_ops_back, check_finite_and_unscale
from ..utils.fp16_lists import AutoMixedPrecisionLists
from six.moves import reduce

__all__ = [
    'distributed_margin_softmax_classify', 'DistributedClassificationOptimizer'
]


class DistributedClassificationOptimizer(Optimizer):
    """
    An optimizer wrapper to generate backward network for distributed
    classification training of model parallelism.
    """

    def __init__(self,
                 optimizer,
                 batch_size,
                 use_fp16=False,
                 fp16_user_dict=None):
        super(DistributedClassificationOptimizer, self).__init__(
            learning_rate=optimizer._learning_rate)
        self._optimizer = optimizer
        self._batch_size = batch_size
        self._use_fp16 = use_fp16

        if self._use_fp16:
            self.init_fp16_params(fp16_user_dict)

    def init_fp16_params(self, fp16_user_dict):
        # set default value for fp16_params_dict
        fp16_params_dict = dict()
        fp16_params_dict['init_loss_scaling'] = 1.0
        fp16_params_dict['incr_every_n_steps'] = 1000
        fp16_params_dict['decr_every_n_nan_or_inf'] = 2
        fp16_params_dict['incr_ratio'] = 2.0
        fp16_params_dict['decr_ratio'] = 0.5
        fp16_params_dict['use_dynamic_loss_scaling'] = True
        fp16_params_dict['amp_lists'] = None
        if fp16_user_dict is not None:
            # update fp16_params_dict
            for key in fp16_user_dict:
                if key in fp16_params_dict:
                    fp16_params_dict[key] = fp16_user_dict[key]
                else:
                    logging.warning(
                        "Can't find name '%s' in our fp16_params_dict. "
                        "Please check your dict key. You can set fp16 params only "
                        "in [init_loss_scaling, incr_every_n_steps, "
                        "decr_every_n_nan_or_inf, incr_ratio, decr_ratio, "
                        "use_dynamic_loss_scaling, amp_lists]" % (key))

        self._amp_lists = fp16_params_dict['amp_lists']
        if self._amp_lists is None:
            self._amp_lists = AutoMixedPrecisionLists()

        self._init_loss_scaling = fp16_params_dict['init_loss_scaling']
        self._loss_scaling = paddle.static.create_global_var(
            name=paddle.utils.unique_name.generate("loss_scaling"),
            shape=[1],
            value=self._init_loss_scaling,
            dtype='float32',
            persistable=True)
        self._use_dynamic_loss_scaling = fp16_params_dict[
            'use_dynamic_loss_scaling']
        if self._use_dynamic_loss_scaling:
            self._incr_every_n_steps = fp16_params_dict['incr_every_n_steps']
            self._decr_every_n_nan_or_inf = fp16_params_dict[
                'decr_every_n_nan_or_inf']
            self._incr_ratio = fp16_params_dict['incr_ratio']
            self._decr_ratio = fp16_params_dict['decr_ratio']
            self._num_good_steps = paddle.static.create_global_var(
                name=paddle.utils.unique_name.generate("num_good_steps"),
                shape=[1],
                value=0,
                dtype='int32',
                persistable=True)
            self._num_bad_steps = paddle.static.create_global_var(
                name=paddle.utils.unique_name.generate("num_bad_steps"),
                shape=[1],
                value=0,
                dtype='int32',
                persistable=True)

        # Ensure the data type of learning rate vars is float32 (same as the
        # master parameter dtype)
        if isinstance(self._optimizer._learning_rate, float):
            self._optimizer._learning_rate_map[paddle.static.default_main_program()] = \
                        paddle.static.create_global_var(
                        name=paddle.utils.unique_name.generate("learning_rate"),
                        shape=[1],
                        value=float(self._optimizer._learning_rate),
                        dtype='float32',
                        persistable=True)

    def fp16_backward(self,
                      block,
                      scalar,
                      startup_program=None,
                      parameter_list=None,
                      no_grad_set=None,
                      callbacks=None):
        rewrite_program(block.program, self._amp_lists)

        if self._use_dynamic_loss_scaling or self._init_loss_scaling != 1.0:
            scaled_scalar = scalar * self._loss_scaling
        else:
            scaled_scalar = scalar

        self._params_grads = self._optimizer.backward(
            scaled_scalar, startup_program, parameter_list, no_grad_set,
            callbacks)

        grads = [g for _, g in self._params_grads]
        with block.program._optimized_guard(grads):
            _, found_inf = check_finite_and_unscale(
                grads, self._loss_scaling, name="find_infinite_scale")
        if self._use_dynamic_loss_scaling:
            with block.program._optimized_guard([]):
                update_loss_scaling(
                    grads,
                    found_inf,
                    self._loss_scaling,
                    self._num_good_steps,
                    self._num_bad_steps,
                    self._incr_every_n_steps,
                    self._decr_every_n_nan_or_inf,
                    self._incr_ratio,
                    self._decr_ratio,
                    name="update_loss_scaling")

        return self._params_grads

    def insert_dist_margin_softmax_backward_op(
            self, block, index, shard_logit, shard_prob, shard_label,
            shard_dim, op_role_key, backward_role, loss_backward_role):
        '''
        during mixed precision training(use_fp16=True), insert backward ops
        '''
        shard_one_hot = block.create_var(
            name=paddle.utils.unique_name.generate('shard_one_hot'),
            dtype=paddle.fluid.core.VarDesc.VarType.FP32)
        # input var of elementwise_add_grad op after scale
        shard_logit_grad_fp32 = block.var(shard_logit.name + '@GRAD')

        if isinstance(shard_dim, int):
            inputs = {'X': shard_label}
            attrs = {
                'depth': shard_dim,
                'allow_out_of_range': True,
                op_role_key: backward_role
            }
        else:
            shard_dim.stop_gradient = True
            inputs = {'X': shard_label, 'depth_tensor': shard_dim}
            attrs = {'allow_out_of_range': True, op_role_key: backward_role}
        block._insert_op(
            index - 2,
            type='one_hot_v2',
            inputs=inputs,
            outputs={'Out': shard_one_hot},
            attrs=attrs)
        block._insert_op(
            index - 1,
            type='elementwise_sub',
            inputs={'X': shard_prob,
                    'Y': shard_one_hot},
            outputs={'Out': shard_logit_grad_fp32},
            attrs={op_role_key: backward_role})
        block._insert_op(
            index,
            type='elementwise_mul',
            inputs={'X': shard_logit_grad_fp32,
                    'Y': self._loss_scaling},
            outputs={'Out': shard_logit_grad_fp32},
            attrs={op_role_key: backward_role})
        block._insert_op(
            index + 1,
            type='scale',
            inputs={'X': shard_logit_grad_fp32},
            outputs={'Out': shard_logit_grad_fp32},
            attrs={
                'scale': 1.0 / self._batch_size,
                op_role_key: loss_backward_role
            })

    def insert_commom_backward_op(self, block, index, shard_logit, shard_prob,
                                  shard_label, shard_dim, op_role_key,
                                  backward_role, loss_backward_role):
        '''
        insert backward ops when not using mixed precision training.
        common use in all lose type.
        '''

        # insert the calculated gradient
        dtype = shard_logit.dtype
        shard_one_hot = block.create_var(
            name=paddle.utils.unique_name.generate('shard_one_hot'),
            dtype=dtype)

        if isinstance(shard_dim, int):
            inputs = {'X': shard_label}
            attrs = {
                'depth': shard_dim,
                'allow_out_of_range': True,
                op_role_key: backward_role
            }
        else:
            shard_dim.stop_gradient = True
            inputs = {'X': shard_label, 'depth_tensor': shard_dim}
            attrs = {'allow_out_of_range': True, op_role_key: backward_role}
        block._insert_op(
            index - 1,
            type='one_hot_v2',
            inputs=inputs,
            outputs={'Out': shard_one_hot},
            attrs=attrs)
        shard_logit_grad = block.var(shard_logit.name + "@GRAD")
        block._insert_op(
            index,
            type='elementwise_sub',
            inputs={'X': shard_prob,
                    'Y': shard_one_hot},
            outputs={'Out': shard_logit_grad},
            attrs={op_role_key: backward_role})
        block._insert_op(
            index + 1,
            type='scale',
            inputs={'X': shard_logit_grad},
            outputs={'Out': shard_logit_grad},
            attrs={
                'scale': 1.0 / self._batch_size,
                op_role_key: loss_backward_role
            })

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None,
                 callbacks=None):
        assert loss._get_info('shard_logit')

        shard_logit = loss._get_info('shard_logit')
        shard_prob = loss._get_info('shard_prob')
        shard_label = loss._get_info('shard_label')
        shard_dim = loss._get_info('shard_dim')

        op_maker = paddle.fluid.core.op_proto_and_checker_maker
        op_role_key = op_maker.kOpRoleAttrName()
        op_role_var_key = op_maker.kOpRoleVarAttrName()
        backward_role = int(op_maker.OpRole.Backward)
        loss_backward_role = int(op_maker.OpRole.Loss) | int(
            op_maker.OpRole.Backward)

        # minimize a scalar of sum to generate the backward network
        scalar = paddle.sum(shard_logit)
        block = loss.block

        if not self._use_fp16:
            ret = self._optimizer.minimize(scalar)

            # remove the unnecessary ops
            index = 0
            for i, op in enumerate(block.ops):
                if op.all_attrs()[op_role_key] == loss_backward_role:
                    index = i
                    break

            assert block.ops[index - 1].type == 'reduce_sum'
            assert block.ops[index].type == 'fill_constant'
            assert block.ops[index + 1].type == 'reduce_sum_grad'
            block._remove_op(index + 1)
            block._remove_op(index)
            block._remove_op(index - 1)

            self.insert_commom_backward_op(
                block, index, shard_logit, shard_prob, shard_label, shard_dim,
                op_role_key, backward_role, loss_backward_role)
            return ret
        else:
            params_grads = self.fp16_backward(block, scalar, startup_program,
                                              parameter_list, no_grad_set,
                                              callbacks)
            index = 0
            for i, op in enumerate(block.ops):
                if op.all_attrs()[op_role_key] == loss_backward_role:
                    index = i
                    break

            assert block.ops[index - 2].type == 'reduce_sum'
            assert block.ops[index - 1].type == 'elementwise_mul'
            assert block.ops[index].type == 'fill_constant'
            assert block.ops[index + 1].type == 'elementwise_mul_grad'
            assert block.ops[index + 2].type == 'reduce_sum_grad'
            assert block.ops[index + 3].type == 'scale'
            assert block.ops[index + 4].type == 'elementwise_add_grad'

            block._remove_op(index + 2)
            block._remove_op(index + 1)
            block._remove_op(index)
            block._remove_op(index - 1)
            block._remove_op(index - 2)

            self.insert_dist_margin_softmax_backward_op(
                block, index, shard_logit, shard_prob, shard_label, shard_dim,
                op_role_key, backward_role, loss_backward_role)

            optimize_ops = self._optimizer.apply_gradients(params_grads)
            ret = optimize_ops, params_grads
            return ret


class DistributedClassifier(object):
    """
    Tookit for distributed classification, in which the parameter of the last
    full-connected layer is distributed to all trainers
    """

    def __init__(self, nclasses, nranks, rank_id, name, sample_ratio=1.0):
        self.nclasses = nclasses
        self.nranks = nranks
        self.rank_id = rank_id
        self.name = name
        self.sample_ratio = sample_ratio

        self.shard_dim = (nclasses + nranks - 1) // nranks
        self.padding_dim = 0
        self.is_equal_division = True
        if nclasses % nranks != 0:
            self.is_equal_division = False
            if rank_id == nranks - 1:
                other_shard_dim = self.shard_dim
                self.shard_dim = nclasses % other_shard_dim
                self.padding_dim = other_shard_dim - self.shard_dim

    def create_parameter(self,
                         dtype,
                         in_dim,
                         param_attr=None,
                         use_bias=True,
                         bias_attr=None,
                         transpose_weight=False):
        if param_attr is None:
            stddev = math.sqrt(2.0 / (in_dim + self.nclasses))
            param_attr = paddle.ParamAttr(
                initializer=paddle.nn.initializer.Normal(std=stddev))
        weight_shape = [self.shard_dim, in_dim
                        ] if transpose_weight else [in_dim, self.shard_dim]
        weight = paddle.static.create_parameter(
            shape=weight_shape,
            dtype=dtype,
            name=self.name,
            attr=param_attr,
            is_bias=False)

        # avoid allreducing gradients for distributed parameters
        weight.is_distributed = True
        # avoid broadcasting distributed parameters in startup program
        paddle.static.default_startup_program().global_block().vars[
            weight.name].is_distributed = True

        bias = None
        if use_bias:
            bias = paddle.static.create_parameter(
                shape=[self.shard_dim],
                attr=bias_attr,
                dtype=dtype,
                is_bias=True)
            # avoid allreducing gradients for distributed parameters
            bias.is_distributed = True
            # avoid broadcasting distributed parameters in startup program
            paddle.static.default_startup_program().global_block().vars[
                bias.name].is_distributed = True
        return weight, bias

    def softmax_with_cross_entropy(self, shard_logit, shard_one_hot):
        shard_max = paddle.max(shard_logit, axis=1, keepdim=True)
        global_max = shard_max
        paddle.distributed.all_reduce(
            global_max, op=paddle.distributed.ReduceOp.MAX)
        shard_logit_new = paddle.subtract(shard_logit, global_max)

        shard_exp = paddle.exp(shard_logit_new)
        shard_demon = paddle.sum(shard_exp, axis=1, keepdim=True)
        global_demon = shard_demon
        paddle.distributed.all_reduce(
            global_demon, op=paddle.distributed.ReduceOp.SUM)

        global_log_demon = paddle.log(global_demon)
        shard_log_prob = shard_logit_new - global_log_demon
        shard_prob = paddle.exp(shard_log_prob)

        target_log_prob = paddle.min(shard_log_prob * shard_one_hot,
                                     axis=1,
                                     keepdim=True)
        shard_loss = paddle.scale(target_log_prob, scale=-1.0)
        #TODO paddle.distributed.reducescatter not found
        global_loss = paddle.fluid.layers.collective._c_reducescatter(
            shard_loss, nranks=self.nranks, use_calc_stream=True)
        return global_loss, shard_prob

    def margin_softmax_classify(self,
                                x,
                                label,
                                margin1=1.0,
                                margin2=0.5,
                                margin3=0.0,
                                logit_scale=64,
                                param_attr=None):
        '''
        reference: ArcFace. https://arxiv.org/abs/1801.07698
        '''
        flatten_dim = reduce(lambda a, b: a * b, x.shape[1:], 1)
        weight, bias = self.create_parameter(
            dtype=x.dtype,
            in_dim=flatten_dim,
            param_attr=param_attr,
            use_bias=False)

        # normalize x
        x_l2 = paddle.sqrt(paddle.sum(paddle.square(x), axis=1, keepdim=True))
        norm_x = paddle.divide(x, x_l2)

        norm_x_list = []
        paddle.distributed.all_gather(norm_x_list, norm_x)
        norm_x_all = paddle.concat(norm_x_list, axis=0)

        label_list = []
        paddle.distributed.all_gather(label_list, label)
        label_all = paddle.concat(label_list, axis=0)
        label_all.stop_gradient = True

        label_all = paddle.reshape(label_all, (-1, 1))
        shard_label = paddle.shard_index(
            label_all,
            index_num=self.nclasses,
            nshards=self.nranks,
            shard_id=self.rank_id,
            ignore_value=-1)
        shard_label = paddle.reshape(shard_label, (-1, ))
        # TODO check necessary
        shard_label.stop_gradient = True

        if self.sample_ratio < 1.0:
            # partial fc sample process
            shard_label, sampled_class_index = class_center_sample(
                shard_label,
                self.shard_dim,
                ratio=self.sample_ratio,
                ignore_label=-1)
            sampled_class_index.stop_gradient = True
            weight = paddle.gather(weight, sampled_class_index, axis=1)
            shard_dim = paddle.shape(sampled_class_index)
        else:
            shard_dim = self.shard_dim

        # normalize weight
        weight_l2 = paddle.sqrt(
            paddle.sum(paddle.square(weight), axis=0, keepdim=True))
        norm_weight = paddle.divide(weight, weight_l2)

        shard_cos = paddle.matmul(norm_x_all, norm_weight)

        theta = paddle.acos(shard_cos)
        if margin1 != 1.0:
            theta = margin1 * theta
        if margin2 != 0.0:
            theta = theta + margin2
        margin_cos = paddle.cos(theta)
        if margin3 != 0.0:
            margin_cos = margin_cos - margin3

        shard_one_hot = paddle.nn.functional.one_hot(
            shard_label, num_classes=shard_dim)
        # TODO check necessary
        shard_one_hot.stop_gradient = True

        diff = paddle.multiply(
            paddle.subtract(margin_cos, shard_cos), shard_one_hot)
        shard_target_cos = paddle.add(shard_cos, diff)
        shard_logit = paddle.scale(shard_target_cos, scale=logit_scale)

        global_loss, shard_prob = self.softmax_with_cross_entropy(
            shard_logit, shard_one_hot)
        avg_loss = paddle.mean(global_loss)

        avg_loss._set_info('shard_logit', shard_logit)
        avg_loss._set_info('shard_prob', shard_prob)
        avg_loss._set_info('shard_label', shard_label)
        avg_loss._set_info('shard_dim', shard_dim)

        return avg_loss


def distributed_margin_softmax_classify(x,
                                        label,
                                        class_num,
                                        nranks,
                                        rank_id,
                                        margin1=1.0,
                                        margin2=0.5,
                                        margin3=0.0,
                                        logit_scale=64.0,
                                        param_attr=None,
                                        sample_ratio=1.0,
                                        name=None):
    """
    Classification layer with margin softmax loss of distibuted version in case of
    too large number of classes. the equation is

    .. math::

        L=-\frac{1}{N}\sum^N_{i=1}\log\frac{e^{s(cos(m_{1}\theta_{y_i}+m_{2})-m_{3})}}{e^{s(cos(m_{1}\theta_{y_i}+m_{2})-m_{3})}+\sum^n_{j=1,j\neq y_i} e^{scos\theta_{y_i}}}

    where the :math: `\theta_{y_i}` is the angle between the feature :math: `x` and
    the representation of class :math: `i`. The details of ArcFace loss
    could be referred to https://arxiv.org/abs/1801.07698.

    Args:
        x (Variable): The feature representation of the input samples. This
            feature will be flattened into 2-D tensor from dimension index
            1. E.g. [32, 1024, 1, 1] will be flattened to [32, 1024].
        label (Variable): The label corresponding to the input samples.
        class_num (integer): The number of classes of the classification problem.
        nranks (integer): The number of ranks of distributed trainers.
        rank_id (integer): The rank index of the current trainer.
        margin1 (float, default 1.0): The angular m1 penalty to enhance
            the intra-class compactness and inter-class discrepancy.
        margin2 (float, default 0.5): The angular m2 penalty to enhance
            the intra-class compactness and inter-class discrepancy.
        margin3 (float, default 0.0): The angular m3 penalty to enhance
            the intra-class compactness and inter-class discrepancy.
        logit_scale (float, default 64.0): The scale factor for logit value
            of cosine range.
        param_attr (ParamAttr, default None): The parameter attribute for
            learnable distributed parameters/weights of this layer.
        sample_ratio (float, default 1.0): sample ratio for partial fc.
            if the sample_ratio less than 1.0, it will use partial fc to training.
        name (str, default None): The name of this layer.
    Returns:
        Variable: The ArcFace loss.


    Examples:
      .. code-block:: python

        #TODO 
    """
    if name is None:
        name = 'dist@margin_softmax@rank@%05d' % rank_id
    classifier = DistributedClassifier(class_num, nranks, rank_id, name,
                                       sample_ratio)
    return classifier.margin_softmax_classify(
        x=x,
        label=label,
        margin1=margin1,
        margin2=margin2,
        margin3=margin3,
        logit_scale=logit_scale,
        param_attr=param_attr)
