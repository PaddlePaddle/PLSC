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

import warnings
import paddle


def _squared_l2_norm(x):
    if x.dtype == paddle.float16:
        square = paddle.square(x)
        sum_square = paddle.sum(square)
        return sum_square

    return paddle._C_ops.squared_l2_norm(x)


class ClipGradByGlobalNorm(object):
    def __init__(self,
                 clip_norm=1.0,
                 clip_norm_max=None,
                 always_clip=False,
                 no_clip_list=[]):
        self.clip_norm = clip_norm
        self.clip_norm_max = clip_norm_max
        self.no_clip_list = no_clip_list
        self.always_clip = always_clip

    def __call__(self, params):
        sum_square_list_fp16 = []
        sum_square_list_fp32 = []
        for param in params:
            if param.grad is None or any(name in param.name
                                         for name in self.no_clip_list):
                continue
            if getattr(param, 'need_clip', True) is False:
                continue
            assert param.grad.dtype in [paddle.float32, paddle.float16]
            sum_square = _squared_l2_norm(param.grad)
            if param.grad.dtype == paddle.float32:
                sum_square_list_fp32.append(sum_square)
            elif param.grad.dtype == paddle.float16:
                sum_square_list_fp16.append(sum_square)

        if len(sum_square_list_fp32) <= 0 and len(sum_square_list_fp16) <= 0:
            warnings.warn('grads_fp32 and grads_fp16 are empty')
            return None

        global_norm_var = []
        if len(sum_square_list_fp16) > 0:
            global_norm_var_fp16 = paddle.add_n(sum_square_list_fp16)
            global_norm_var.append(global_norm_var_fp16.astype("float32"))
        if len(sum_square_list_fp32) > 0:
            global_norm_var_fp32 = paddle.add_n(sum_square_list_fp32)
            global_norm_var.append(global_norm_var_fp32)

        global_norm = paddle.add_n(global_norm_var)
        global_norm = paddle.sqrt(global_norm)

        if not self.always_clip and global_norm <= self.clip_norm:
            return

        clip_coef_fp32 = self.clip_norm / (global_norm + 1e-6)
        if self.clip_norm_max is not None:
            clip_coef_fp32 = paddle.clip(
                clip_coef_fp32, max=self.clip_norm_max)

        for param in params:
            if param.grad is None or any(name in param.name
                                         for name in self.no_clip_list):
                continue
            if getattr(param, 'need_clip', True) is False:
                continue

            clip_coef = clip_coef_fp32
            if param.grad.dtype == paddle.float16:
                clip_coef = clip_coef_fp32.astype("float16")

            # inplace calculate
            paddle.fluid.framework._dygraph_tracer().trace_op(
                type="elementwise_mul",
                inputs={'X': param.grad,
                        'Y': clip_coef},
                outputs={'Out': param.grad},
                attrs={'axis': -1})
