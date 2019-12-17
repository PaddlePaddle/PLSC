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

from __future__ import print_function
import math

from six.moves import reduce
import paddle.fluid as fluid
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.framework import Variable, default_startup_program
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Normal, Constant
import paddle.fluid.layers.nn as nn
import paddle.fluid.layers.ops as ops
import paddle.fluid.layers as layers
import paddle.fluid.layers.collective as collective
from paddle.fluid.optimizer import Optimizer
import paddle.fluid.unique_name as unique_name
from ..utils.fp16_utils import rewrite_program, update_role_var_grad
from ..utils.fp16_utils import update_loss_scaling, move_optimize_ops_back
from ..utils.fp16_lists import AutoMixedPrecisionLists


class DistributedClassificationOptimizer(Optimizer):
    '''
    A optimizer wrapper to generate backward network for distributed
    classification training of model parallelism.
    '''

    def __init__(self,
                 optimizer,
                 batch_size,
                 use_fp16=False,
                 loss_type='dist_arcface',
                 amp_lists=None,
                 init_loss_scaling=1.0,
                 incr_every_n_steps=1000,
                 decr_every_n_nan_or_inf=2,
                 incr_ratio=2.0,
                 decr_ratio=0.5,
                 use_dynamic_loss_scaling=True):
        super(DistributedClassificationOptimizer, self).__init__(
            learning_rate=0.1)

        self._optimizer = optimizer
        self._batch_size = batch_size
        self._use_fp16 = use_fp16

        self._amp_lists = amp_lists
        if amp_lists is None:
            self._amp_lists = AutoMixedPrecisionLists()

        self._param_grads = None
        self._scaled_loss = None
        self._loss_type = loss_type
        self._init_loss_scaling = init_loss_scaling
        self._loss_scaling = layers.create_global_var(
            name=unique_name.generate("loss_scaling"),
            shape=[1],
            value=init_loss_scaling,
            dtype='float32',
            persistable=True)
        self._use_dynamic_loss_scaling = use_dynamic_loss_scaling
        if self._use_dynamic_loss_scaling:
            self._incr_every_n_steps = layers.fill_constant(
                shape=[1], dtype='int32', value=incr_every_n_steps)
            self._decr_every_n_nan_or_inf = layers.fill_constant(
                shape=[1], dtype='int32', value=decr_every_n_nan_or_inf)
            self._incr_ratio = incr_ratio
            self._decr_ratio = decr_ratio
            self._num_good_steps = layers.create_global_var(
                name=unique_name.generate("num_good_steps"),
                shape=[1],
                value=0,
                dtype='int32',
                persistable=True)
            self._num_bad_steps = layers.create_global_var(
                name=unique_name.generate("num_bad_steps"),
                shape=[1],
                value=0,
                dtype='int32',
                persistable=True)

        # Ensure the data type of learning rate vars is float32 (same as the
        # master parameter dtype)
        if isinstance(optimizer._learning_rate, float):
            optimizer._learning_rate_map[fluid.default_main_program()] = \
                        layers.create_global_var(
                        name=unique_name.generate("learning_rate"),
                        shape=[1],
                        value=float(optimizer._learning_rate),
                        dtype='float32',
                        persistable=True)

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

        op_maker = fluid.core.op_proto_and_checker_maker
        op_role_key = op_maker.kOpRoleAttrName()
        op_role_var_key = op_maker.kOpRoleVarAttrName()
        backward_role = int(op_maker.OpRole.Backward)
        loss_backward_role = int(op_maker.OpRole.Loss) | int(
            op_maker.OpRole.Backward)

        # minimize a scalar of reduce_sum to generate the backward network
        scalar = fluid.layers.reduce_sum(shard_logit)

        if not self._use_fp16:
            ret = self._optimizer.minimize(scalar)

            block = loss.block
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

            # insert the calculated gradient
            dtype = shard_logit.dtype
            shard_one_hot = fluid.layers.create_tensor(
                dtype, name='shard_one_hot')
            block._insert_op(
                index - 1,
                type='one_hot',
                inputs={'X': shard_label},
                outputs={'Out': shard_one_hot},
                attrs={
                    'depth': shard_dim,
                    'allow_out_of_range': True,
                    op_role_key: backward_role
                })
            shard_logit_grad = fluid.layers.create_tensor(dtype,
                name=fluid.backward._append_grad_suffix_(shard_logit.name))
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
            return ret
        else:
            block = loss.block
            rewrite_program(block.program, self._amp_lists)
            self._params_grads = self._optimizer.backward(
                scalar, startup_program, parameter_list,
                no_grad_set, callbacks)
            update_role_var_grad(block.program, self._params_grads)
            move_optimize_ops_back(block.program.global_block())
            scaled_params_grads = []
            for p, g in self._params_grads:
                with fluid.default_main_program()._optimized_guard([p, g]):
                    scaled_g = g / self._loss_scaling
                    scaled_params_grads.append([p, scaled_g])

            index = 0
            for i, op in enumerate(block.ops):
                if op.all_attrs()[op_role_key] == loss_backward_role:
                    index = i
                    break
            fp32 = fluid.core.VarDesc.VarType.FP32
            dtype = shard_logit.dtype

            if self._loss_type == 'dist_arcface':
                assert block.ops[index - 2].type == 'fill_constant'
                assert block.ops[index - 1].type == 'reduce_sum'
                assert block.ops[index].type == 'fill_constant'
                assert block.ops[index + 1].type == 'reduce_sum_grad'
                assert block.ops[index + 2].type == 'scale'
                assert block.ops[index + 3].type == 'elementwise_add_grad'

                block._remove_op(index + 2)
                block._remove_op(index + 1)
                block._remove_op(index)
                block._remove_op(index - 1)

                # insert the calculated gradient
                shard_one_hot = fluid.layers.create_tensor(
                    dtype, name='shard_one_hot')
                block._insert_op(
                    index - 1,
                    type='one_hot',
                    inputs={'X': shard_label},
                    outputs={'Out': shard_one_hot},
                    attrs={
                        'depth': shard_dim,
                        'allow_out_of_range': True,
                        op_role_key: backward_role
                    })
                shard_one_hot_fp32 = fluid.layers.create_tensor(
                    fp32, name=(shard_one_hot.name+".cast_fp32"))
                block._insert_op(
                    index,
                    type="cast",
                    inputs={"X": shard_one_hot},
                    outputs={"Out": shard_one_hot_fp32},
                    attrs={
                        "in_dtype": fluid.core.VarDesc.VarType.FP16,
                        "out_dtype": fluid.core.VarDesc.VarType.FP32,
                        op_role_key: backward_role
                    })
                name = 'tmp_3@GRAD'
                shard_logit_grad_fp32 = block.var(name)

                block._insert_op(
                    index+1,
                    type='elementwise_sub',
                    inputs={'X': shard_prob,
                            'Y': shard_one_hot_fp32},
                    outputs={'Out': shard_logit_grad_fp32},
                    attrs={op_role_key: backward_role})

                block._insert_op(
                    index+2,
                    type='elementwise_mul',
                    inputs={'X': shard_logit_grad_fp32,
                            'Y': self._loss_scaling},
                    outputs={'Out': shard_logit_grad_fp32},
                    attrs={op_role_key: backward_role})

                block._insert_op(
                    index+3,
                    type='scale',
                    inputs={'X': shard_logit_grad_fp32},
                    outputs={'Out': shard_logit_grad_fp32},
                    attrs={
                        'scale': 1.0 / self._batch_size,
                        op_role_key: loss_backward_role
                    })
            elif self._loss_type == 'dist_softmax':
                assert block.ops[index - 1].type == 'reduce_sum'
                assert block.ops[index].type == 'fill_constant'
                assert block.ops[index + 1].type == 'reduce_sum_grad'
                assert block.ops[index + 2].type == 'cast'
                assert block.ops[index + 3].type == 'elementwise_add_grad'
 
                block._remove_op(index + 1)
                block._remove_op(index)
                block._remove_op(index - 1)

                # insert the calculated gradient 
                shard_one_hot = fluid.layers.create_tensor(
                    fp32, name='shard_one_hot')
                shard_one_hot_fp32 = fluid.layers.create_tensor(fp32, 
                    name=(shard_one_hot.name+".cast_fp32"))
                shard_logit_grad_fp32 = block.var(
                    shard_logit.name+".cast_fp32@GRAD")
                block._insert_op(
                    index - 1,
                    type='one_hot',
                    inputs={'X': shard_label},
                    outputs={'Out': shard_one_hot_fp32},
                    attrs={
                        'depth': shard_dim,
                        'allow_out_of_range': True,
                        op_role_key: backward_role
                    })
                
                block._insert_op(
                    index,
                    type='elementwise_sub',
                    inputs={'X': shard_prob,
                            'Y': shard_one_hot_fp32},
                    outputs={'Out': shard_logit_grad_fp32},
                    attrs={op_role_key: backward_role})
                block._insert_op(
                    index + 1,
                    type='elementwise_mul',
                    inputs={'X': shard_logit_grad_fp32,
                            'Y': self._loss_scaling},
                    outputs={'Out': shard_logit_grad_fp32},
                    attrs={op_role_key: backward_role})
                block._insert_op(
                    index + 2,
                    type='scale',
                    inputs={'X': shard_logit_grad_fp32},
                    outputs={'Out': shard_logit_grad_fp32},
                    attrs={
                        'scale': 1.0 / self._batch_size,
                        op_role_key: loss_backward_role
                    })

            if self._use_dynamic_loss_scaling:
                grads = [layers.reduce_sum(g) for [_, g] in scaled_params_grads]
                all_grads = layers.concat(grads)
                all_grads_sum = layers.reduce_sum(all_grads)
                is_overall_finite = layers.isfinite(all_grads_sum)

                update_loss_scaling(is_overall_finite, self._loss_scaling,
                                    self._num_good_steps, self._num_bad_steps,
                                    self._incr_every_n_steps,
                                    self._decr_every_n_nan_or_inf,
                                    self._incr_ratio,
                                    self._decr_ratio)

                with layers.Switch() as switch:
                    with switch.case(is_overall_finite):
                        pass
                    with switch.default():
                        for _, g in scaled_params_grads:
                            layers.assign(layers.zeros_like(g), g)

            optimize_ops = self._optimizer.apply_gradients(scaled_params_grads)
            ret = optimize_ops, scaled_params_grads

        return ret


class DistributedClassifier(object):
    '''
    Tookit for distributed classification, in which the parameter of the last
    full-connected layer is distributed to all trainers
    '''

    def __init__(self, nclasses, nranks, rank_id, layer_helper):
        self.nclasses = nclasses
        self.nranks = nranks
        self.rank_id = rank_id
        self._layer_helper = layer_helper

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
                         bias_attr=None,
                         transpose_weight=False,
                         use_bias=True):
        if param_attr is None:
            stdv = math.sqrt(2.0 / (in_dim + self.nclasses))
            param_attr = ParamAttr(initializer=Normal(scale=stdv))
        weight_shape = [self.shard_dim, in_dim
                        ] if transpose_weight else [in_dim, self.shard_dim]
        weight = self._layer_helper.create_parameter(
            shape=weight_shape, dtype=dtype, attr=param_attr, is_bias=False)
        # avoid distributed parameter allreduce gradients
        weight.is_distributed = True
        # avoid distributed parameter broadcasting in startup program
        default_startup_program().global_block().vars[
            weight.name].is_distributed = True

        bias = None
        if use_bias:
            bias = self._layer_helper.create_parameter(
                shape=[self.shard_dim],
                attr=bias_attr,
                dtype=dtype,
                is_bias=True)
            bias.is_distributed = True
            default_startup_program().global_block().vars[
                bias.name].is_distributed = True
        return weight, bias

    def softmax_with_cross_entropy(self, shard_logit, shard_label):
        shard_max = nn.reduce_max(shard_logit, dim=1, keep_dim=True)
        global_max = collective._c_allreduce(
            shard_max, reduce_type='max', use_calc_stream=True)
        shard_logit_new = nn.elementwise_sub(shard_logit, global_max)

        shard_exp = ops.exp(shard_logit_new)
        shard_demon = nn.reduce_sum(shard_exp, dim=1, keep_dim=True)
        global_demon = collective._c_allreduce(
            shard_demon, reduce_type='sum', use_calc_stream=True)

        global_log_demon = nn.log(global_demon)
        shard_log_prob = shard_logit_new - global_log_demon
        shard_prob = ops.exp(shard_log_prob)

        shard_one_hot = nn.one_hot(
            shard_label, depth=self.shard_dim, allow_out_of_range=True)
        target_log_prob = nn.reduce_min(
            shard_log_prob * shard_one_hot, dim=1, keep_dim=True)
        shard_loss = nn.scale(target_log_prob, scale=-1.0)
        global_loss = collective._c_reducescatter(
            shard_loss, nranks=self.nranks, use_calc_stream=True)
        return global_loss, shard_prob

    def softmax_classify(self,
                         x,
                         label,
                         param_attr=None,
                         use_bias=True,
                         bias_attr=None):
        flatten_dim = reduce(lambda a, b: a * b, x.shape[1:], 1)
        weight, bias = self.create_parameter(
            dtype=x.dtype,
            in_dim=flatten_dim,
            param_attr=param_attr,
            bias_attr=bias_attr,
            use_bias=use_bias)

        x_all = collective._c_allgather(
            x, nranks=self.nranks, use_calc_stream=True)
        label_all = collective._c_allgather(
            label, nranks=self.nranks, use_calc_stream=True)
        label_all.stop_gradient = True

        shard_fc = nn.mul(x_all, weight, x_num_col_dims=1)
        if use_bias:
            shard_fc = nn.elementwise_add(shard_fc, bias)

        shard_label = nn.shard_index(
            label_all,
            index_num=self.nclasses,
            nshards=self.nranks,
            shard_id=self.rank_id,
            ignore_value=-1)
        shard_label.stop_gradient = True

        global_loss, shard_prob = self.softmax_with_cross_entropy(shard_fc,
                                                                  shard_label)
        avg_loss = nn.mean(global_loss)

        avg_loss._set_info('shard_logit', shard_fc)
        avg_loss._set_info('shard_prob', shard_prob)
        avg_loss._set_info('shard_label', shard_label)
        avg_loss._set_info('shard_dim', self.shard_dim)

        return avg_loss

    def arcface_classify(self,
                         x,
                         label,
                         margin=0.5,
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
        x_l2 = ops.sqrt(nn.reduce_sum(ops.square(x), dim=1))
        norm_x = nn.elementwise_div(x, x_l2, axis=0)

        norm_x_all = collective._c_allgather(
            norm_x, nranks=self.nranks, use_calc_stream=True)
        label_all = collective._c_allgather(
            label, nranks=self.nranks, use_calc_stream=True)
        label_all.stop_gradient = True
        shard_label = nn.shard_index(
            label_all,
            index_num=self.nclasses,
            nshards=self.nranks,
            shard_id=self.rank_id,
            ignore_value=-1)
        # TODO check necessary
        shard_label.stop_gradient = True

        # normalize weight
        weight_l2 = ops.sqrt(nn.reduce_sum(ops.square(weight), dim=0))
        norm_weight = nn.elementwise_div(weight, weight_l2, axis=1)

        shard_cos = nn.mul(norm_x_all, norm_weight, x_num_col_dims=1)

        theta = ops.acos(shard_cos)
        margin_cos = ops.cos(theta + margin)

        shard_one_hot = nn.one_hot(
            shard_label, depth=self.shard_dim, allow_out_of_range=True)
        # TODO check necessary
        shard_one_hot.stop_gradient = True

        diff = (margin_cos - shard_cos) * shard_one_hot
        shard_target_cos = shard_cos + diff
        shard_logit = nn.scale(shard_target_cos, scale=logit_scale)

        global_loss, shard_prob = self.softmax_with_cross_entropy(shard_logit,
                                                                  shard_label)
        avg_loss = nn.mean(global_loss)

        avg_loss._set_info('shard_logit', shard_logit)
        avg_loss._set_info('shard_prob', shard_prob)
        avg_loss._set_info('shard_label', shard_label)
        avg_loss._set_info('shard_dim', self.shard_dim)

        return avg_loss


def _distributed_softmax_classify(x,
                                  label,
                                  class_num,
                                  nranks,
                                  rank_id,
                                  param_attr=None,
                                  use_bias=True,
                                  bias_attr=None,
                                  name=None):
    '''
    Classification layer with FC, softmax and cross entropy calculation of
    distibuted version in case of too large number of classes.
    
    Args:
        x (Variable): The feature representation of the input samples. This
            feature will be flattened into 2-D tensor from dimension index
            1. E.g. [32, 1024, 1, 1] will be flattened to [32, 1024].
        label (Variable): The label corresponding to the input samples.
        class_num (integer): The number of classes of the classification problem.
        nranks (integer): The number of ranks of distributed trainers.
        rank_id (integer): The rank index of the current trainer.
        param_attr (ParamAttr, default None): The parameter attribute for
            learnable distributed parameters/weights of this layer.
        use_bias (float, default 64.0): The scale factor for logit value
            of cosine range.
        name (str, default None): The name of this layer.
    Returns:
        Variable: The ArcFace loss.


    Examples:
      .. code-block:: python

        import paddle.fluid as fluid
        input = fluid.layers.data(name="input",
                                  shape=[32, 1024], 
                                  dtype='float32', 
                                  append_batch_size=False)                   
        label = fluid.layers.data(name="label",
                                  shape=[32, 1], 
                                  dtype='int64', 
                                  append_batch_size=False)                   
        y = fluid.layers.collective.distributed_softmax_classify(x=input,
                                                            label=label,
                                                            class_num=1000,
                                                            nranks=8,
                                                            rank_id=0)
    '''

    if name is None:
        name = 'dist@softmax@rank@%05d' % rank_id
    helper = LayerHelper(name, **locals())
    classifier = DistributedClassifier(class_num, nranks, rank_id, helper)
    return classifier.softmax_classify(x, label, param_attr, use_bias,
                                       bias_attr)


def _distributed_arcface_classify(x,
                                  label,
                                  class_num,
                                  nranks,
                                  rank_id,
                                  margin=0.5,
                                  logit_scale=64.0,
                                  param_attr=None,
                                  name=None):
    '''
    Classification layer with ArcFace loss of distibuted version in case of
    too large number of classes. the equation is

    .. math::

        L=-\frac{1}{N}\sum^N_{i=1}\log\frac{e^{s(cos(\theta_{y_i}+m))}}{e^{s(cos(\theta_{y_i}+m))}+\sum^n_{j=1,j\neq y_i} e^{scos\theta_{y_i}}}

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
        margin (float, default 0.5): The angular margin penalty to enhance
            the intra-class compactness and inter-class discrepancy.
        logit_scale (float, default 64.0): The scale factor for logit value
            of cosine range.
        param_attr (ParamAttr, default None): The parameter attribute for
            learnable distributed parameters/weights of this layer.
        name (str, default None): The name of this layer.
    Returns:
        Variable: The ArcFace loss.


    Examples:
      .. code-block:: python

        import paddle.fluid as fluid
        input = fluid.layers.data(name="input",
                                  shape=[32, 1024], 
                                  dtype='float32', 
                                  append_batch_size=False)                   
        label = fluid.layers.data(name="label",
                                  shape=[32, 1], 
                                  dtype='int64', 
                                  append_batch_size=False)                   
        y = fluid.layers.collective.distributed_arcface_classify(x=input,
                                                                 label=label,
                                                                 class_num=1000,
                                                                 nranks=8,
                                                                 rank_id=0)
    '''
    if name is None:
        name = 'dist@arcface@rank@%05d' % rank_id
    helper = LayerHelper(name, **locals())
    classifier = DistributedClassifier(class_num, nranks, rank_id, helper)
    return classifier.arcface_classify(
        x=x,
        label=label,
        margin=margin,
        logit_scale=logit_scale,
        param_attr=param_attr)

