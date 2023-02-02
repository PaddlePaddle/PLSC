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
from collections import Iterable
import paddle
import paddle.nn as nn

from plsc.loss.celoss import ViTCELoss, CELoss
from plsc.loss.distill_loss import MSELoss


class MTLoss(nn.Layer):
    """
    multi-task loss framework
    """

    def __init__(self, task_names, losses, weights=1.0):
        super().__init__()
        self.loss_func = {}
        self.task_names = task_names
        self.instance_losses(losses)
        self.loss_weight = {}
        if isinstance(weights, float):
            weights = len(self.loss_func) * [weights]
        assert len(self.loss_func) == len(
            weights), "Length of loss_func should be equal to weights"
        weight_sum = sum(weights)
        for task_id in self.loss_func:
            self.loss_weight[task_id] = weights[task_id] / weight_sum

    def instance_losses(self, losses):
        assert isinstance(losses, Iterable) and len(losses) > 0, \
            "losses should be iterable and length greater than 0"
        self.loss_func = {}
        self.loss_names = {}

        for loss_item in losses:
            assert isinstance(loss_item, dict) and len(loss_item.keys()) == 1, \
                "item in losses should be config dict whose length is one(loss class config)"
            name = list(loss_item.keys())[0]
            params = loss_item[name]
            task_ids = params.pop("task_ids", [0])
            if not isinstance(task_ids, list):
                task_ids = [task_ids]
            for task_id in task_ids:
                self.loss_func[task_id] = eval(name)(**params)
                self.loss_names[task_id] = name

    @staticmethod
    def cast_fp32(input):
        if input.dtype != paddle.float32:
            input = paddle.cast(input, 'float32')
        return input

    def __call__(self, input, target):
        # target: [label, task]
        loss_dict = {}
        total_loss = 0.0
        assert isinstance(
            target, dict), "target shold be a dict including keys(label, task)"
        label = target["label"]
        task = target["task"]
        for idx in self.loss_func:
            cond = task == idx
            logits = input[self.task_names[idx]][cond]
            if isinstance(label, dict):
                if self.task_names[idx] in label:
                    labels = label[self.task_names[idx]][cond]
                else:
                    print("label should be a tensor, not dict")
            else:
                labels = label[cond]
            loss = self.loss_func[idx](logits, labels)
            loss_dict[idx] = loss[self.loss_names[idx]]
            total_loss += loss_dict[idx] * self.loss_weight[idx]
        return loss_dict, total_loss
