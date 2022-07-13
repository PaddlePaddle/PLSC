# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class MarginLoss(nn.Layer):
    """
    SphereFace: https://arxiv.org/abs/1704.08063
    m1=1.35, m2=0.0, m3=0.0, s=64.0
    CosFace: https://arxiv.org/abs/1801.09414
    m1=1.0, m2=0.0, m3=0.4, s=64.0
    ArcFace: https://arxiv.org/abs/1801.07698
    m1=1.0, m2=0.5, m3=0.0, s=64.0
    
    Default: ArcFace
    """

    def __init__(self, m1=1.0, m2=0.5, m3=0.0, s=64.0, model_parallel=False):
        super().__init__()
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.s = s

        # Default we use model parallel when group=None.
        # When group=False, it is equal to data parallel.
        self.group = None
        if not model_parallel:
            self.group = False

    def forward(self, x, label):
        if isinstance(x, dict):
            x = x["logits"]

        loss = F.margin_cross_entropy(
            x,
            label,
            margin1=self.m1,
            margin2=self.m2,
            margin3=self.m3,
            scale=self.s,
            return_softmax=False,
            reduction=None,
            group=self.group, )

        loss = loss.mean()
        return {"MarginLoss": loss}
