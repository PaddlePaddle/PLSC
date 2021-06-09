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

from easydict import EasyDict as edict
"""
Default Parameters
"""

config = edict()

config.train_batch_size = 128
config.test_batch_size = 120
config.val_targets = 'lfw'
config.dataset_dir = './train_data'
config.train_image_num = 5822653
config.model_name = 'ResNet50'
config.train_epochs = None
config.train_steps = 180000
config.checkpoint_dir = ""
config.with_test = True
config.model_save_dir = "output"
config.warmup_epochs = 0
config.model_parallel = False

config.loss_type = "arcface"
config.num_classes = 85742
config.sample_ratio = 1.0
config.image_shape = (3, 112, 112)
config.margin1 = 1.0
config.margin2 = 0.5
config.margin3 = 0.0
config.scale = 64.0
config.lr = 0.1
config.lr_steps = (100000, 160000, 180000)
config.emb_dim = 512
