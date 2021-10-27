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


@paddle.no_grad()
def sync_params(parameters):
    for param in parameters:
        paddle.distributed.broadcast(
            param.detach(), src=0, group=None, use_calc_stream=True)


@paddle.no_grad()
def sync_gradients(parameters):
    grad_var_set_fp32 = set()
    grad_vars_fp32 = []
    grad_var_set_fp16 = set()
    grad_vars_fp16 = []

    p_size_fp32 = 0
    p_size_fp16 = 0
    for param in parameters:
        if param.trainable and (param._grad_ivar() is not None):
            g_var = param._grad_ivar()
            assert not g_var._is_sparse(
            ), "Now, it doesn't support sparse parameters"
            assert g_var.dtype in [paddle.float32, paddle.float16]
            if g_var.dtype == paddle.float32:
                grad_vars_fp32.append(g_var)
                assert g_var not in grad_var_set_fp32
                grad_var_set_fp32.add(g_var)

            else:
                grad_vars_fp16.append(g_var)
                assert g_var not in grad_var_set_fp16
                grad_var_set_fp16.add(g_var)

    if len(grad_vars_fp32) > 0:
        coalesced_grads_and_vars_fp32 = \
            paddle.fluid.dygraph.parallel.build_groups(grad_vars_fp32, 128 * 1024 * 1024)
        for coalesced_grad, _, _ in coalesced_grads_and_vars_fp32:
            paddle.distributed.all_reduce(coalesced_grad)
        paddle.fluid.dygraph.parallel._split_tensors(coalesced_grads_and_vars_fp32)

    if len(grad_vars_fp16) > 0:
        coalesced_grads_and_vars_fp16 = \
            paddle.fluid.dygraph.parallel.build_groups(grad_vars_fp16, 128 * 1024 * 1024)
        for coalesced_grad, _, _ in coalesced_grads_and_vars_fp16:
            paddle.distributed.all_reduce(coalesced_grad)
        paddle.fluid.dygraph.parallel._split_tensors(coalesced_grads_and_vars_fp16)
