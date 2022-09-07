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

from __future__ import print_function

import numpy as np
import os

import paddle

from .common_dataset import CommonDataset
from plsc.utils import logger


class FaceIdentificationDataset(CommonDataset):
    def __init__(self,
                 image_root,
                 cls_label_path,
                 transform_ops=None,
                 delimiter="\t"):
        super(FaceIdentificationDataset, self).__init__(
            image_root, cls_label_path, transform_ops, delimiter)

    def _load_anno(self, seed=None):
        assert os.path.exists(
            self._cls_path), f"{self._cls_path} does not exists"
        assert os.path.exists(
            self._img_root), f"{self._img_root} does not exists"
        self.images = []
        self.labels = []

        logger.info('Loading dataset {}'.format(self._cls_path))
        with open(self._cls_path) as fd:
            lines = fd.readlines()
            if seed is not None:
                np.random.RandomState(seed).shuffle(lines)
            for l in lines:
                l = l.strip().split(self.delimiter)
                self.images.append(os.path.join(self._img_root, l[0]))
                self.labels.append(np.int32(l[1]))
                assert os.path.exists(self.images[-1])
        logger.info('Load dataset finished, {} samples'.format(
            len(self.images)))


class FaceVerificationDataset(CommonDataset):
    def __init__(self,
                 image_root,
                 cls_label_path,
                 transform_ops=None,
                 delimiter="\t"):
        super(FaceVerificationDataset, self).__init__(
            image_root, cls_label_path, transform_ops, delimiter)

    def _load_anno(self, seed=None):
        assert os.path.exists(
            self._cls_path), f"{self._cls_path} does not exists"
        assert os.path.exists(
            self._img_root), f"{self._img_root} does not exists"
        self.images = []
        self.labels = []

        with open(self._cls_path) as fd:
            lines = fd.readlines()
            for l in lines:
                l = l.strip().split(self.delimiter)
                self.images.append(os.path.join(self._img_root, l[0]))
                self.images.append(os.path.join(self._img_root, l[1]))
                self.labels.append(np.int32(l[2]))
                self.labels.append(np.int32(l[2]))
                assert os.path.exists(self.images[-2])
                assert os.path.exists(self.images[-1])


class FaceRandomDataset(paddle.io.Dataset):
    def __init__(self, num_classes):
        super(FaceRandomDataset, self).__init__()
        self.num_classes = num_classes
        self.label_list = np.random.randint(
            0, num_classes, (51200, ), dtype=np.int32)

        self.total_num_samples = len(self.label_list)
        self.num_samples = len(self.label_list)

    def __getitem__(self, idx):
        label = self.label_list[idx]
        img = np.random.uniform(
            low=-1.0, high=1.0, size=(3, 112, 112)).astype(np.float32)

        label = np.int32(label)

        return img, label

    def __len__(self):
        return self.num_samples
