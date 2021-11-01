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

from collections import defaultdict
from paddle.amp import GradScaler
from paddle import _C_ops
import paddle

@paddle.no_grad()
def clip_grad_norm_(grads_fp32, grads_fp16, grad_norm_clip=2.0):
    
    norm_fp32 = paddle.sum(paddle.stack([paddle.sum(g.detach() ** 2) for g in grads_fp32]))
    if len(grads_fp16) > 0:
        norm_fp16 = paddle.sum(paddle.stack([paddle.sum(paddle.cast(g.detach(), 'float32') ** 2) for g in grads_fp16]))
        global_norm = paddle.sqrt(norm_fp32 + paddle.cast(norm_fp16, 'float32'))
    else:
        global_norm = paddle.sqrt(norm_fp32)
    
    clip_coef_fp32 = paddle.clip(grad_norm_clip / (global_norm + 1e-6), max=2.0)
    grads_fp32 = [g.scale_(clip_coef_fp32) for g in grads_fp32]
    
    if len(grads_fp16) > 0:
        clip_coef_fp16 = paddle.cast(clip_coef_fp32, 'float16')
        grads_fp16 = [g.scale_(clip_coef_fp16) for g in grads_fp16]
    
    return global_norm

class LSCGradScaler(GradScaler):
    def __init__(self,
                 enable=True,
                 init_loss_scaling=2.**15,
                 incr_ratio=2.0,
                 decr_ratio=0.5,
                 incr_every_n_steps=1000,
                 decr_every_n_nan_or_inf=2,
                 use_dynamic_loss_scaling=True):
        super(LSCGradScaler, self).__init__(
            enable, init_loss_scaling, incr_ratio, decr_ratio,
            incr_every_n_steps, decr_every_n_nan_or_inf,
            use_dynamic_loss_scaling)

    def step(self, optimizer, classifier=None):
        if not self._enable:
            if classifier is not None:
                classifier.step(optimizer)
            return optimizer.step()

        # unscale the grad
        self._unscale(optimizer)

        if self._found_inf:
            self._cache_founf_inf = True
        else:
            optimizer.step()
            if classifier is not None:
                classifier.step(optimizer)

            self._cache_founf_inf = False

        if self._use_dynamic_loss_scaling:
            # update the scale
            self._update()
            
    @paddle.no_grad()
    def _unscale(self, optimizer):
        if not self._enable:
            return

        param_grads_dict = defaultdict(list)
        dist_param_grads_dict = defaultdict(list)
        if getattr(optimizer, '_param_groups', None) and isinstance(
                optimizer._param_groups[0], dict):
            for group in optimizer._param_groups:
                for param in group['params']:
                    if not param.is_distributed:
                        if param._grad_ivar() is not None:
                            param_grads_dict[param._grad_ivar().dtype].append(
                                param._grad_ivar())
                    else:
                        if param._grad_ivar() is not None:
                            dist_param_grads_dict[param._grad_ivar(
                            ).dtype].append(param._grad_ivar())
        else:
            for param in optimizer._parameter_list:
                if not param.is_distributed:
                    if param._grad_ivar() is not None:
                        param_grads_dict[param._grad_ivar().dtype].append(
                            param._grad_ivar())
                else:
                    if param._grad_ivar() is not None:
                        dist_param_grads_dict[param._grad_ivar().dtype].append(
                            param._grad_ivar())
        for dtype in dist_param_grads_dict:
            for grad in dist_param_grads_dict[dtype]:
                self._found_inf = paddle.logical_not(
                    paddle.all(paddle.isfinite(grad)))
                if self._found_inf:
                    print('Found inf or nan in classifier, dtype is', dtype)
                    return
                
        grads_fp32 = []
        grads_fp16 = []
        if len(param_grads_dict[paddle.float32]) > 0:
            coalesced_grads_and_vars_fp32 = \
                paddle.fluid.dygraph.parallel.build_groups(param_grads_dict[paddle.float32], 128 * 1024 * 1024)
            for coalesced_grad, _, _ in coalesced_grads_and_vars_fp32: 
                paddle.distributed.all_reduce(coalesced_grad)
                grads_fp32.append(coalesced_grad)

            _C_ops.check_finite_and_unscale(grads_fp32, self._scale,
                                            grads_fp32, self._found_inf)
            if self._found_inf:
                print('Found inf or nan in backbone, dtype is', paddle.float32)
                return

        if len(param_grads_dict[paddle.float16]) > 0:
            coalesced_grads_and_vars_fp16 = \
                paddle.fluid.dygraph.parallel.build_groups(param_grads_dict[paddle.float16], 128 * 1024 * 1024)
            for coalesced_grad, _, _ in coalesced_grads_and_vars_fp16:
                paddle.distributed.all_reduce(coalesced_grad)
                grads_fp16.append(coalesced_grad)

            _C_ops.check_finite_and_unscale(grads_fp16, self._scale,
                                            grads_fp16, self._found_inf)
            if self._found_inf:
                print('Found inf or nan in backbone, dtype is', paddle.float16)
                return

        clip_grad_norm_(grads_fp32, grads_fp16)
        if len(param_grads_dict[paddle.float16]) > 0:
            paddle.fluid.dygraph.parallel._split_tensors(coalesced_grads_and_vars_fp16)
        if len(param_grads_dict[paddle.float32]) > 0:
            paddle.fluid.dygraph.parallel._split_tensors(coalesced_grads_and_vars_fp32)
