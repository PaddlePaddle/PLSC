# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy

import paddle
import paddle.nn as nn

from plsc.loss.celoss import ViTCELoss, CELoss
from plsc.loss.distill_loss import MSELoss


class MTLoss(nn.Layer):
    """
    multi-task loss framework
    """

    def __init__(self, task_names, cfg):
        super().__init__()
        self.loss_func = {}
        self.loss_weight = {}
        self.task_names = task_names
        self.loss_names = {}
        assert isinstance(cfg, list), "operator config should be a list"
        for loss_item in cfg:
            # more than 1 loss func, 1 loss func has more than 1 tasks
            assert isinstance(loss_item, dict), "yaml format error"
            loss_name = list(loss_item.keys())[0]
            param = loss_item[loss_name]
            task_id_list = param.pop("task_ids", [0])  # default task 0
            assert "weight" in param, \
                "weight must be in param, but param just contains {}".format(
                param.keys())
            loss_weight = param.pop("weight", 1.0)
            if isinstance(loss_weight, float):
                loss_weight = len(task_id_list) * [loss_weight]
            assert len(loss_weight) == len(task_id_list), \
                "task weights length must be equal to task number"
            weight_sum = sum(loss_weight)
            for task_id in task_id_list:
                self.loss_names[task_id] = loss_name
                self.loss_weight[task_id] = loss_weight[task_id] / weight_sum
                self.loss_func[task_id] = eval(loss_name)(**param)

    @staticmethod
    def cast_fp32(input):
        if input.dtype != paddle.float32:
            input = paddle.cast(input, 'float32')
        return input

    def __call__(self, input, target, task):
        loss_dict = {}
        total_loss = 0.0
        for idx in self.loss_func:
            cond = task == idx
            logits = input[self.task_names[idx]][cond]
            if isinstance(target, dict):
                labels = target[self.task_names[idx]][cond]
            else:
                labels = target[cond]
            loss = self.loss_func[idx](logits, labels)
            weight = self.loss_weight[idx]
            loss_dict[idx] = loss[self.loss_names[idx]]
            total_loss += loss_dict[idx] * weight
        return loss_dict, total_loss
