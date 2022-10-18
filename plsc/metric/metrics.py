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

import numpy as np
import sklearn
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from . import lfw_utils

__all__ = ['TopkAcc', 'mAP', 'LFWAcc']


class TopkAcc(nn.Layer):
    def __init__(self, topk=(1, 5)):
        super().__init__()
        assert isinstance(topk, (int, list, tuple))
        if isinstance(topk, int):
            topk = [topk]
        self.topk = topk

    def forward(self, x, label):
        if isinstance(x, dict):
            x = x["logits"]

        if len(label.shape) == 1:
            label = label.reshape([label.shape[0], -1])

        if label.dtype == paddle.int32:
            label = paddle.cast(label, 'int64')
        metric_dict = dict()
        for i, k in enumerate(self.topk):
            acc = paddle.metric.accuracy(x, label, k=k).item()
            metric_dict["top{}".format(k)] = acc
            if i == 0:
                metric_dict["metric"] = acc

        return metric_dict


class mAP(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, similarities_matrix, query_img_id, gallery_img_id,
                keep_mask):
        metric_dict = dict()

        choosen_indices = paddle.argsort(
            similarities_matrix, axis=1, descending=True)
        gallery_labels_transpose = paddle.transpose(gallery_img_id, [1, 0])
        gallery_labels_transpose = paddle.broadcast_to(
            gallery_labels_transpose,
            shape=[
                choosen_indices.shape[0], gallery_labels_transpose.shape[1]
            ])
        choosen_label = paddle.index_sample(gallery_labels_transpose,
                                            choosen_indices)
        equal_flag = paddle.equal(choosen_label, query_img_id)
        if keep_mask is not None:
            keep_mask = paddle.index_sample(
                keep_mask.astype('float32'), choosen_indices)
            equal_flag = paddle.logical_and(equal_flag,
                                            keep_mask.astype('bool'))
        equal_flag = paddle.cast(equal_flag, 'float32')

        num_rel = paddle.sum(equal_flag, axis=1)
        num_rel = paddle.greater_than(num_rel, paddle.to_tensor(0.))
        num_rel_index = paddle.nonzero(num_rel.astype("int"))
        num_rel_index = paddle.reshape(num_rel_index, [num_rel_index.shape[0]])
        equal_flag = paddle.index_select(equal_flag, num_rel_index, axis=0)

        acc_sum = paddle.cumsum(equal_flag, axis=1)
        div = paddle.arange(acc_sum.shape[1]).astype("float32") + 1
        precision = paddle.divide(acc_sum, div)

        #calc map
        precision_mask = paddle.multiply(equal_flag, precision)
        ap = paddle.sum(precision_mask, axis=1) / paddle.sum(equal_flag,
                                                             axis=1)
        metric_dict["mAP"] = paddle.mean(ap).item()
        metric_dict["metric"] = metric_dict["mAP"]

        return metric_dict


class LFWAcc(nn.Layer):
    def __init__(self, flip_test=True, nrof_folds=10, pca=0):
        super().__init__()
        self.flip_test = flip_test
        self.nrof_folds = nrof_folds
        self.pca = pca

    def forward(self, embeddings, actual_issame):

        metric_dict = dict()

        embeddings = np.asarray(embeddings)
        actual_issame = np.asarray(actual_issame)

        _xnorm = 0.0
        _xnorm_cnt = 0
        for i in range(embeddings.shape[0]):
            _em = embeddings[i]
            _norm = np.linalg.norm(_em)
            _xnorm += _norm
            _xnorm_cnt += 1
        _xnorm /= _xnorm_cnt

        if self.flip_test:
            num_embed = embeddings.shape[0]
            num_label = actual_issame.shape[0]
            assert num_embed == num_label, "The number of features must be equal to labels when flip_test=True"
            assert num_embed % 2 == 0, "The number of features must divide 2 when flip_test=True"
            embeddings = embeddings[:num_embed // 2] + embeddings[num_embed //
                                                                  2:]
            actual_issame = actual_issame[:num_embed // 2]

        embeddings = sklearn.preprocessing.normalize(embeddings)

        # Calculate evaluation metrics
        thresholds = np.arange(0, 4, 0.01)
        embeddings1 = embeddings[0::2]
        embeddings2 = embeddings[1::2]
        actual_issame = actual_issame[0::2]
        tpr, fpr, accuracy = lfw_utils.calculate_roc(
            thresholds,
            embeddings1,
            embeddings2,
            actual_issame,
            nrof_folds=self.nrof_folds,
            pca=self.pca)

        acc2, std2 = np.mean(accuracy), np.std(accuracy)

        metric_dict["std"] = std2
        metric_dict["xnorm"] = _xnorm
        if self.flip_test:
            metric_dict["accuracy-flip"] = acc2
        else:
            metric_dict["accuracy"] = acc2
        metric_dict["metric"] = acc2

        return metric_dict
