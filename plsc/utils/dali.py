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


class HybridValPipe(Pipeline):
    """
    Create validate pipe line.
    """
    def __init__(self,
                 file_root,
                 file_list,
                 batch_size,
                 mean,
                 std,
                 device_id,
                 shard_id=0,
                 eii=None,
                 num_shards=1,
                 random_shuffle=False,
                 num_threads=4,
                 seed=42,
                 data_layout='NCHW'):
        super(HybridRectValPipe, self).__init__(batch_size,
                                                num_threads,
                                                device_id,
                                                seed=seed)
        self.size_iter = iter(eii)
        #self.input = ops.FileReader(file_root=file_root,
        #                            file_list=file_list,
        #                            shard_id=shard_id,
        #                            num_shards=num_shards,
        #                            random_shuffle=random_shuffle)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        shorter_size = int(crop * 1.14)
        self.res = ops.Resize(device="gpu",
                              resize_shorter=shorter_size,
                              interp_type=interp)
        output_layout = convert_data_layout(data_layout)
        self.input_jpeg = ops.ExternalSource()
        self.input_label = ops.ExternalSource()
        self.output_w = ops.ExternalSource()
        self.output_h = ops.ExternalSource()
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=output_layout,
                                            image_type=types.RGB,
                                            mean=mean,
                                            std=std)
        self.to_int64 = ops.Cast(dtype=types.INT64, device="gpu")

    def define_graph(self):
        #jpegs, labels = self.input(name="Reader")
        self.jpeg = self.input_jpeg()
        self.label = self.input_label()
        images = self.decode(self.jpeg)
        images = self.res(images)
        self.size_w = self.output_w()
        self.size_h = self.output_h()
        output = self.cmnp(images, crop_w=self.size_w, crop_h=self.size_h)
        return [output, self.to_int64(self.label.gpu())]

    def iter_setup(self):
        try:
            (jpegs, labels, ws, hs) = self.size_iter.next()
            self.feed_input(self.jpeg, jpegs)
            self.feed_input(self.label, labels)
            self.feed_input(self.size_w, ws)
            self.feed_input(self.size_h, hs)
        except StopIteration:
            raise StopIteration

    def __len__(self):
        #return self.epoch_size("Reader")
        return 50000


def build(batch_size,
          data_dir,
          epoch_id,
          file_list=None,
          mode='train',
          trainer_id=None,
          trainers_num=None,
          gpu_id=0,
          data_layout='NCHW',
          eii=None):
    env = os.environ
    assert float(env.get('FLAGS_fraction_of_gpu_memory_to_use', 0.92)) < 0.9, \
        "Please leave enough GPU memory for DALI workspace, e.g., by setting" \
        " `export FLAGS_fraction_of_gpu_memory_to_use=0.8`"
    print("batch_size:", batch_size)

    mean = [127.5, 127.5, 127.5]
    std =  [128.0, 128.0, 128.0]

    if file_list:
        file_list = os.path.join(data_dir, file_list)
        if not os.path.exists(file_list):
            raise ValueError("{} does not exist in {}.".format(
                              file_list, file_root))
    if mode != 'train':
        raise ValueError("Only support train mode now.")

    assert trainer_id is not None and trainers_num is not None, \
        "Please set trainer_id and trainers_num."
    print("dali gpu_id: {}, shard_id: {}, num_shards: {}".format(
                                                                 gpu_id,
                                                                 trainer_id,
                                                                 trainers_num))
    pipe = HybridTrainPipe(data_dir,
                           file_list,
                           batch_size,
                           mean,
                           std,
                           device_id=gpu_id,
                           shard_id=trainer_id,
                           num_shards=trainers_num,
                           seed=epoch_id,
                           data_layout=data_layout,
                           num_threads=4)
    pipe.build()
    pipelines = [pipe]
    sample_per_shard = len(pipe) // trainers_num
    
    return DALIGenericIterator(pipelines,
                               ['image', 'label'],
                               size=sample_per_shard)


def train(batch_size,
          data_dir,
          file_list=None,
          trainer_id=None,
          trainers_num=None,
          gpu_id=0,
          eii=None,
          data_layout="NCHW"):
    return build(batch_size,
                 data_dir,
                 file_list,
                 'train',
                 trainer_id=trainer_id,
                 trainers_num=trainers_num,
                 gpu_id=gpu_id,
                 eii=eii,
                 data_layout=data_layout)


def val(settings,
        trainer_id=None,
        trainers_num=None,
        gpu_id=0,
        data_layout="NCHW",
        rect_val=False,
        eii=None):
    return build(settings,
                 'val',
                 trainer_id=trainer_id,
                 trainers_num=trainers_num,
                 gpu_id=gpu_id,
                 data_layout=data_layout,
                 rect_val=rect_val,
                 eii=eii)


class ExternalSizeIterator(object):
    def __init__(self, batch_size, target_size, val_dir, val_image_num=50000):
        idx_ar_sorted = sort_ar(val_dir)
        idx_sorted, _ = zip(*idx_ar_sorted)
        idx2ar = map_idx2ar(idx_ar_sorted, batch_size)
        self.idx_sorted = idx_sorted
        self.idx2ar = idx2ar
        self.batch_size = batch_size
        self.target_size = target_size
        self.val_dir = val_dir

        file_list = []
        label_list = []
        filename = os.path.join(val_dir, "val_list.txt")
        with open(filename, 'r') as f:
            for line in f.xreadlines():
                line = line.strip().split(' ')
                file = line[0]
                label = int(line[1])
                file = os.path.join(val_dir, file)
                file_list.append(file)
                label_list.append(label)
        self.file_list = file_list
        self.label_list = label_list

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        width = []
        height = []
        jpegs = []
        labels = []
        for _ in range(self.batch_size):
            if self.i >= 50000:
                if len(width) > 0:
                    return (jpegs, labels, weight, height)
                raise StopIteration
            idx = self.idx_sorted[self.i]
            filename = self.file_list[idx]
            f = open(filename, 'rb')
            jpegs.append(np.frombuffer(f.read(), dtype=np.uint8))
            label = self.label_list[idx]
            labels.append(np.array([label], dtype=np.uint64))
            target_ar = self.idx2ar[idx]
            self.i += 1
            if target_ar < 1:
                h = int(self.target_size / target_ar)
                h = h // 8 * 8
                w = self.target_size
                width.append(np.array(w, dtype=np.int32))
                height.append(np.array(h, dtype=np.int32))
            else:
                w = int(self.target_size * target_ar)
                w = w // 8 * 8
                h = self.target_size
                weight.append(np.array(w, dtype=np.int32))
                height.append(np.array(h, dtype=np.int32))
        return (jpegs, labels, weight, height)

    next = __next__

