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

import paddle
from paddle.fluid import core
from paddle.fluid import framework
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.regularizer import L2DecayRegularizer
from collections import defaultdict


class HybridOptimizer(paddle.optimizer.Momentum):
    def __init__(self,
                 learning_rate=0.001,
                 momentum=0.9,
                 parameters=None,
                 use_nesterov=False,
                 weight_decay=None,
                 grad_clip=None,
                 multi_precision=False,
                 rescale_grad=1.0,
                 name=None):
        super(HybridOptimizer, self).__init__(
            learning_rate, momentum, parameters, use_nesterov, weight_decay,
            grad_clip, multi_precision, rescale_grad, name)
        # learning_rate must be float32, regardless of whether the precsion is used.
        self._dtype = 'float32'

    def _get_accumulator(self, name, param):
        if self._name is not None:
            name = self._name + "_" + name
        if name not in self._accumulators:
            raise Exception("Accumulator {} does not exist".format(name))
        if param.name not in self._accumulators[name]:
            if getattr(param, 'is_sparse_grad', None) and param.stop_gradient:
                self._add_accumulator(self._velocity_acc_str, param)
            else:
                raise Exception(
                    "Accumulator {} does not exist for parameter {}".format(
                        name, param.name))
        return self._accumulators[name][param.name]

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)
        if isinstance(param_and_grad, dict):
            param_and_grad = self._update_param_group(param_and_grad)

        velocity_acc = self._get_accumulator(self._velocity_acc_str,
                                             param_and_grad[0])
        lr = self._create_param_lr(param_and_grad)

        # For fusion of momentum and l2decay 
        param = param_and_grad[0]
        regularization_method = self._regularization_method
        regularization_coeff = self._regularization_coeff
        if hasattr(param, 'regularizer'):
            # we skip param's l2decay before, so fuse it with momentum here.
            if isinstance(param.regularizer, L2DecayRegularizer):
                regularization_method = "l2_decay"
                regularization_coeff = param.regularizer._regularization_coeff
            # the param's regularization has been done before, we avoid do l2decay in momentum.
            elif param.regularizer is not None:
                regularization_method = ""
                regularization_coeff = 0

        find_master = self._multi_precision and param_and_grad[
            0].dtype == core.VarDesc.VarType.FP16
        master_weight = (self._master_weights[param_and_grad[0].name]
                         if find_master else None)

        if isinstance(param_and_grad, dict):
            self._update_regularization(param_and_grad['weight_decay'])

        if getattr(param_and_grad[0], 'is_sparse_grad', None):
            index = getattr(param_and_grad[0], 'index', None)
            axis = getattr(param_and_grad[0], 'axis', None)
            _, _ = paddle._C_ops.sparse_momentum(
                param_and_grad[0], param_and_grad[1], velocity_acc, index, lr,
                param_and_grad[0], velocity_acc, 'mu', self._momentum,
                'use_nesterov', self._use_nesterov, 'regularization_method',
                self._regularization_method, 'regularization_coeff',
                self._regularization_coeff, 'axis', axis)
        else:
            _, _, _ = paddle._C_ops.momentum(
                param_and_grad[0], param_and_grad[1], velocity_acc, lr,
                master_weight, param_and_grad[0], velocity_acc, master_weight,
                'mu', self._momentum, 'use_nesterov', self._use_nesterov,
                'regularization_method', regularization_method,
                'regularization_coeff', regularization_coeff,
                'multi_precision', find_master)

        return None

    @paddle.no_grad()
    def step(self):
        if not isinstance(self._param_groups[0], dict):
            params_grads = []
            for param in self._param_groups:
                if param._grad_ivar() is not None:
                    grad_var = param._grad_ivar()
                    params_grads.append((param, grad_var))
                elif getattr(param, 'is_sparse_grad', None) \
                        and getattr(param, 'sparse_grad', None) is not None:
                    grad_var = getattr(param, 'sparse_grad')
                    params_grads.append((param, grad_var))

            self._apply_optimize(
                loss=None, startup_program=None, params_grads=params_grads)

        else:
            # optimize parameters in groups
            for param_group in self._param_groups:
                params_grads = defaultdict(lambda: list())
                for param in param_group['params']:
                    if param._grad_ivar() is not None:
                        grad_var = param._grad_ivar()
                        params_grads['params'].append((param, grad_var))
                    elif getattr(param, 'is_sparse_grad', None) \
                            and getattr(param, 'sparse_grad', None) is not None:
                        grad_var = getattr(param, 'sparse_grad')
                        params_grads['params'].append((param, grad_var))

                params_grads.update(
                    {k: v
                     for k, v in param_group.items() if k != 'params'})
                self._apply_optimize(
                    loss=None, startup_program=None, params_grads=params_grads)

    def clear_grad(self):
        if self._parameter_list is None or not isinstance(
                self._parameter_list[0], dict):
            for p in self._parameter_list:
                if not p.stop_gradient:
                    p.clear_gradient()

                if getattr(p, 'is_sparse_grad', None):
                    delattr(p, 'sparse_grad')
                    delattr(p, 'index')
                    delattr(p, 'axis')
        else:
            for param_group in self._param_groups:
                for p in param_group['params']:
                    if not p.stop_gradient:
                        p.clear_gradient()

                    if getattr(p, 'is_sparse_grad', None):
                        delattr(p, 'sparse_grad')
                        delattr(p, 'index')
                        delattr(p, 'axis')

    def _create_optimization_pass(self, parameters_and_grads):
        global_block = framework.default_main_program().global_block()
        target_block = global_block
        current_block = framework.default_main_program().current_block()
        if current_block.idx != global_block.idx:
            assert current_block.backward_block_idx != -1, \
                "current block is not global_block, but it doesn't have backward block."
            target_block = framework.default_main_program().blocks[
                current_block.backward_block_idx]

        start = len(target_block.ops)
        self.helper = LayerHelper(self.__class__.__name__)
        params_grads_device_map = parameters_and_grads['params'] if isinstance(
            parameters_and_grads, dict) else parameters_and_grads
        self._update_param_device_map(params_grads_device_map, target_block)
        if isinstance(parameters_and_grads, list):
            self._create_accumulators(
                target_block,
                [p[0] for p in parameters_and_grads if not p[0].stop_gradient \
                    or getattr(p[0], 'is_sparse_grad', None)])

        else:
            params_acc_dict = parameters_and_grads.copy()
            params_acc_dict['params'] = [
                p[0] for p in params_acc_dict['params']
                if not p[0].stop_gradient or getattr(p[0], 'is_sparse_grad',
                                                     None)
            ]
            self._create_accumulators(target_block, params_acc_dict)

        self._create_global_learning_rate()

        if isinstance(parameters_and_grads, list):
            for param_and_grad in parameters_and_grads:
                if param_and_grad[1] is None:
                    continue
                if param_and_grad[0].stop_gradient is False or \
                        getattr(param_and_grad[0], 'is_sparse_grad', None):
                    self._append_optimize_op(target_block, param_and_grad)
        else:
            for param_and_grad in parameters_and_grads['params']:
                if param_and_grad[1] is None:
                    continue
                if param_and_grad[0].stop_gradient is False or \
                        getattr(param_and_grad[0], 'is_sparse_grad', None):
                    param_grad_dict = dict()
                    param_grad_dict['params'] = param_and_grad
                    param_grad_dict.update({
                        k: v
                        for k, v in parameters_and_grads.items()
                        if k != 'params'
                    })
                    self._append_optimize_op(target_block, param_grad_dict)

        # Get custom finish ops for subclasses
        # FIXME: Need to fix this once we figure out how to handle dependencies
        self._finish_update(target_block, parameters_and_grads)

        end = len(target_block.ops)
        return target_block._slice_ops(start, end)
