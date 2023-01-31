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
Single-task Dataset and ConcatDataset are realized.
Multi-task dataset(ConcatDataset) can be composed by multiple single-task datasets.
"""

import warnings
import bisect
import cv2
from os.path import join
import numpy as np
import random

import paddle
from paddle.io import Dataset
from plsc.data.utils import create_preprocess_operators


class SingleTaskDataset(Dataset):
    """
    Single-task Dataset.
    The input file includes single task dataset.
    """

    def __init__(self, task_id, data_root, label_path, transform_ops):
        self.task_id = task_id
        self.data_root = data_root
        self.transform_ops = None
        if transform_ops is not None:
            self.transform_ops = create_preprocess_operators(transform_ops)
        self.data_list = []
        with open(join(data_root, label_path), "r") as f:
            for line in f:
                img_path, label = line.strip().split(" ")
                self.data_list.append(
                    (join(data_root, "images", img_path), int(label)))

    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]
        with open(img_path, 'rb') as f:
            img = f.read()
        if self.transform_ops:
            img = self.transform_ops(img)
        if label == -1:
            label = 0
        label = paddle.to_tensor(label, dtype=paddle.int32)
        return img, label, self.task_id

    def __len__(self):
        return len(self.data_list)


class ConcatDataset(Dataset):
    """

    Dataset that are composed by multiple datasets.
    Multi-task Dataset can be the concatenation of single-task datasets.
    """

    @staticmethod
    def cumsum(sequence, ratio_list):
        r, s = [], 0
        for i, e in enumerate(sequence):
            l = int(len(e) * ratio_list[i])
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets, dataset_ratio=None):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)

        if dataset_ratio is not None:
            assert len(dataset_ratio) == len(datasets)
            self.dataset_ratio = {
                i: dataset_ratio[i]
                for i in range(len(dataset_ratio))
            }
        else:
            self.dataset_ratio = {i: 1. for i in range(len(datasets))}

        self.cumulative_sizes = self.cumsum(self.datasets, self.dataset_ratio)
        self.idx_ds_map = {
            idx: bisect.bisect_right(self.cumulative_sizes, idx)
            for idx in range(self.__len__())
        }

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = self.idx_ds_map[idx]
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        if sample_idx >= len(self.datasets[dataset_idx]):
            sample_idx = random.choice(range(len(self.datasets[dataset_idx])))
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn(
            "cummulative_sizes attribute is renamed to "
            "cumulative_sizes",
            DeprecationWarning,
            stacklevel=2)
        return self.cumulative_sizes


class MultiTaskDataset(Dataset):
    """
    Multi-Task Dataset.
    The input file includes multi-task datasets.
    """

    def __init__(self, task_ids, data_root, label_path, transform_ops):
        """

        Args:
            task_ids: task id list
            data_root:
            label_path:
            transform_ops:
        """
        self.task_ids = task_ids
        self.data_root = data_root
        self.transform_ops = None
        if transform_ops is not None:
            self.transform_ops = create_preprocess_operators(transform_ops)
        self.data_list = []
        with open(join(data_root, label_path), "r") as f:
            for line in f:
                img_path, labels = line.strip().split(" ", 1)
                labels = [int(label) for label in labels.strip().split(" ")]
                self.data_list.append(
                    (join(data_root, "images", img_path), labels))

    def __getitem__(self, idx):
        img_path, labels = self.data_list[idx]
        with open(img_path, 'rb') as f:
            img = f.read()
        if self.transform_ops:
            img = self.transform_ops(img)
        labels = [0 if label == -1 else label for label in labels]
        labels = paddle.to_tensor(np.array(labels), dtype=paddle.int32)
        return img, labels, np.array(self.task_ids)

    def __len__(self):
        return len(self.data_list)
