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
"""
Multi-task Model Framework is realized.
Combine backbone and multiple encoder layers in this framework.
"""

import os
from typing import List, Dict
from collections import OrderedDict

import paddle
from paddle.nn import Layer, LayerDict, LayerList
from plsc.models.base_model import Model
from plsc.models.multi_task.ResNet_backbone import *
from plsc.models.multi_task.head import *


class MTLModel(Model):
    """
    Multi-task Model Framework.
    Recomputing can be turned on.
    """

    def __init__(self, task_names, backbone, heads):
        """

        Args:
            task_names: task name list
            backbone: backbone for feature extraction (or config dict)
            encoder_heads: Dict<task_names: Layer> (or config list)
            recompute_on: if recompute is used
            recompute_params: recompute layers
        """
        super(MTLModel, self).__init__()
        self.task_names = task_names

        if isinstance(backbone, Model):
            self.backbone = backbone
        else:
            self.backbone = self.instances_from_cfg(backbone)
        # {task_names: Layer}
        heads = self.instances_from_cfg(heads)
        self.heads = LayerDict(sublayers=heads)

    def instances_from_cfg(self, cfg):
        # instantiate layer from config dict
        if isinstance(cfg, dict):
            name = cfg.pop("name", None)
            if name is not None:
                try:
                    module = eval(name)(**cfg)
                except Exception as e:
                    print("instance cfg error: ", e)
                else:
                    return module
        if isinstance(cfg, list):
            module_dic = {}
            for item in cfg:
                if isinstance(item, dict) and len(item) == 1:
                    name = list(item.keys())[0]
                    params = item[name]
                    task_ids = params.pop("task_ids", None)
                    class_nums = params.pop("class_nums", None)
                    if task_ids and class_nums:
                        for task_id, class_num in zip(task_ids, class_nums):
                            module = eval(name)(class_num=class_num, **params)
                            module_dic[self.task_names[task_id]] = module
            if len(module_dic) > 0:
                return module_dic
        return None

    def forward(self, inputs, output_task_names=None):
        output = {}
        features = self.backbone(inputs)
        if output_task_names is not None:
            for task_name in output_task_names:
                output[task_name] = self.heads[task_name](features)
        else:
            for task_name in self.heads:
                output[task_name] = self.heads[task_name](features)
        return output

    def save(self, path, local_rank=0, rank=0):
        # save model
        dist_state_dict = OrderedDict()
        state_dict = self.state_dict()
        for name, param in list(state_dict.items()):
            if param.is_distributed:
                dist_state_dict[name] = state_dict.pop(name)

        if local_rank == 0:
            paddle.save(state_dict, path + ".pdparams")

        if len(dist_state_dict) > 0:
            paddle.save(dist_state_dict,
                        path + "_rank{}.pdparams".format(rank))

    def load_pretrained(self, path, rank=0, finetune=False):
        # load pretrained model
        if not os.path.exists(path + '.pdparams'):
            raise ValueError("Model pretrain path {} does not "
                             "exists.".format(path))

        state_dict = paddle.load(path + ".pdparams")

        dist_param_path = path + "_rank{}.pdparams".format(rank)
        if os.path.exists(dist_param_path):
            dist_state_dict = paddle.load(dist_param_path)
            state_dict.update(dist_state_dict)
            # clear
            dist_state_dict.clear()

        if not finetune:
            self.set_dict(state_dict)
