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
from plsc.models.layers.base_model import Model
from plsc.core.recompute import wrap_forward, recompute_forward


class MTLModel(Model):
    """
    Multi-task Model Framework.
    Recomputing can be turned on.
    """

    def __init__(self,
                 backbone: Layer,
                 encoder_heads: Dict,
                 recompute_on=False,
                 recompute_params=None):
        """

        Args:
            backbone: backbone for feature extraction
            encoder_heads: Dict<task_names: Layer>
            recompute_on: if recompute is used
            recompute_params: recompute layers
        """
        super(MTLModel, self).__init__()
        if recompute_params is None:
            recompute_params = {}
        self.backbone = backbone
        # {task_names: Layer}
        self.encoder_heads = LayerDict(sublayers=encoder_heads)
        self.recompute_on = recompute_on
        if self.recompute_on:
            self.recompute_warp(self.backbone, **recompute_params)
            for task_name in self.encoder_heads:
                self.recompute_warp(self.encoder_heads[task_name],
                                    **recompute_params)

    def recompute_warp(self,
                       model,
                       layer_interval=1,
                       names=[],
                       exclude_names=None):
        # recompute layers in names list or use layer_interval setting,
        # layers in excluded names are excluded.
        if exclude_names is None:
            exclude_names = ["Dropout", "dropout", "pool"]
        for idx, (name, layer) in enumerate(model._sub_layers.items()):
            if name in exclude_names:
                print(f"continue: {name}")
                continue
            if isinstance(layer, paddle.nn.LayerList):
                for i, (name, sub_layer) in enumerate(layer.named_sublayers()):
                    if name in exclude_names:
                        print(f"continue: {name}")
                        continue
                    if layer_interval >= 1 and idx % layer_interval == 0:
                        print('recompute: ', name)
                        sub_layer.forward = wrap_forward(sub_layer.forward,
                                                         recompute_forward)
            else:
                if layer_interval >= 1 and idx % layer_interval == 0:
                    print('recompute: ', name)
                    layer.forward = wrap_forward(layer.forward,
                                                 recompute_forward)

    def forward(self, inputs, output_task_names=None):
        output = {}
        features = self.backbone(inputs)
        if output_task_names is not None:
            for task_name in output_task_names:
                output[task_name] = self.encoder_heads[task_name](features)
        else:
            for task_name in self.encoder_heads:
                output[task_name] = self.encoder_heads[task_name](features)
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
