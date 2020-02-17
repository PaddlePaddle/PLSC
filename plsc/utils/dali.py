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

from __future__ import division
import os
import math
import pickle
import numpy as np
import random
import time

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.paddle import DALIGenericIterator

import paddle
from paddle import fluid


def convert_data_layout(data_layout):
    if data_layout == 'NCHW':
        return types.NCHW
    elif data_layout == 'NHWC':
        return types.NHWC
    else:
        raise ValueError("Not supported data_layout: {}".format(data_layout))


class HybridTrainPipe(Pipeline):
    """
    Create training pipeline.
    For more information please refer:
    https://docs.nvidia.com/deeplearning/sdk/dali-master-branch-user-guide/docs/plugins/paddle_tutorials.html
    Note: You may need to find the newest DALI version.
    """
    def __init__(self,
                 file_root,
                 file_list,
                 batch_size,
                 mean,
                 std,
                 device_id,
                 shard_id=0,
                 num_shards=1,
                 random_shuffle=True,
                 num_threads=4,
                 seed=42,
                 data_layout="NCHW"):
        super(HybridTrainPipe, self).__init__(batch_size,
                                              num_threads,
                                              device_id,
                                              seed=seed,
                                              prefetch_queue_depth=8)
        self.input = ops.FileReader(file_root=file_root,
                                    file_list=file_list,
                                    shard_id=shard_id,
                                    num_shards=num_shards,
                                    random_shuffle=random_shuffle)
        device_memory_padding = 211025920
        host_memory_padding = 140544512
        self.decode = ops.ImageDecoder(
                                    device='mixed',
                                    output_type=types.RGB,
                                    device_memory_padding=device_memory_padding,
                                    host_memory_padding=host_memory_padding)
        output_layout = convert_data_layout(data_layout)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=output_layout,
                                            image_type=types.RGB,
                                            mean=mean,
                                            std=std)
        self.coin = ops.CoinFlip(probability=0.5)
        self.to_int64 = ops.Cast(dtype=types.INT64, device="gpu")

    def define_graph(self):
        rng = self.coin()
        jpegs, labels = self.input(name="Reader")
        images = self.decode(jpegs)
        output = self.cmnp(images.gpu(), mirror=rng)
        return [output, self.to_int64(labels.gpu())]

    def __len__(self):
        return self.epoch_size("Reader")


def build(batch_size,
          data_dir,
          file_list=None,
          mode='train',
          trainer_id=0,
          trainers_num=1,
          mean=[127.5, 127.5, 127.5],
          std=[128.0, 128.0, 128.0],
          gpu_id=0,
          seed=42,
          num_threads=4,
          data_layout='NCHW'):
    env = os.environ
    assert float(env.get('FLAGS_fraction_of_gpu_memory_to_use', 0.92)) < 0.9, \
        "Please leave enough GPU memory for DALI workspace, e.g., by setting" \
        " `export FLAGS_fraction_of_gpu_memory_to_use=0.8`"


    if file_list:
        file_list = os.path.join(data_dir, file_list)
        if not os.path.exists(file_list):
            raise ValueError("{} does not exist in {}.".format(
                              file_list, data_dir))
    if mode != 'train':
        raise ValueError("Only train mode is supported now.")

    assert trainer_id is not None and trainers_num is not None, \
        "Please set trainer_id and trainers_num."
    print("dali gpu_id: {}, shard_id: {}, num_shards: {}".format(
                                                                 gpu_id,
                                                                 trainer_id,
                                                                 trainers_num))
    pipe = HybridTrainPipe(data_dir,
                           file_list,
                           batch_size,
                           mean=mean,
                           std=std,
                           device_id=gpu_id,
                           shard_id=trainer_id,
                           num_shards=trainers_num,
                           seed=seed,
                           data_layout=data_layout,
                           num_threads=num_threads)
    pipe.build()
    pipelines = [pipe]
    sample_per_shard = len(pipe) // trainers_num
    
    return DALIGenericIterator(pipelines,
                               ['image', 'label'],
                               size=sample_per_shard)


def train(batch_size,
          data_dir,
          file_list=None,
          trainer_id=0,
          trainers_num=1,
          mean=[127.5, 127.5, 127.5],
          std=[128.0, 128.0, 128.0],
          gpu_id=0,
          num_threads=4,
          seed=None,
          data_layout="NCHW"):
    if seed is None:
        seed = int(time.time())
    return build(batch_size,
                 data_dir,
                 file_list=file_list,
                 seed=seed,
                 mode='train',
                 mean=mean,
                 std=std,
                 trainer_id=trainer_id,
                 trainers_num=trainers_num,
                 gpu_id=gpu_id,
                 num_threads=num_threads,
                 data_layout=data_layout)

