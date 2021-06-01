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

import functools
import math
import os
import pickle
import random

import numpy as np
import paddle
import six
from PIL import Image, ImageEnhance

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
from io import BytesIO

random.seed(0)

DATA_DIM = 112
THREAD = 8
BUF_SIZE = 10240

#TEST_LIST = 'lfw,cfp_fp,agedb_30,cfp_ff'
TEST_LIST = 'lfw'


def get_train_image_list(data_dir):
    train_list_file = os.path.join(data_dir, 'label.txt')
    train_list = open(train_list_file, "r").readlines()
    random.shuffle(train_list)
    train_image_list = []
    trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
    trainer_count = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
    per_node_lines = (len(train_list) + trainer_count - 1) // trainer_count
    train_list += train_list[0:per_node_lines * trainer_count - len(
        train_list)]
    lines = train_list[trainer_id * per_node_lines:(trainer_id + 1) *
                       per_node_lines]
    print("read images from %d, length: %d, total: %d" %
          (trainer_id * per_node_lines, per_node_lines, len(train_list)))

    for i, item in enumerate(lines):
        path, label = item.strip().split()
        label = int(label)
        train_image_list.append((path, label))
    print("train_data size:", len(train_image_list))
    return train_image_list


img_mean = np.array([127.5, 127.5, 127.5]).reshape((3, 1, 1))
img_std = np.array([128.0, 128.0, 128.0]).reshape((3, 1, 1))


def resize_short(img, target_size):
    percent = float(target_size) / min(img.size[0], img.size[1])
    resized_width = int(round(img.size[0] * percent))
    resized_height = int(round(img.size[1] * percent))
    img = img.resize((resized_width, resized_height), Image.BILINEAR)
    return img


def scale(img, size):
    w, h = img.size
    if (w <= h and w == size) or (h <= w and h == size):
        return img
    if w < h:
        ow = size
        oh = int(size * h / w)
        return img.resize((ow, oh), Image.BILINEAR)
    else:
        oh = size
        ow = int(size * w / h)
        return img.resize((ow, oh), Image.BILINEAR)


def center_crop(img, size):
    w, h = img.size
    th, tw = int(size), int(size)
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))
    return img.crop((x1, y1, x1 + tw, y1 + th))


def crop_image(img, target_size, center):
    width, height = img.size
    size = target_size
    if center == True:
        w_start = (width - size) / 2
        h_start = (height - size) / 2
    else:
        w_start = random.randint(0, width - size)
        h_start = random.randint(0, height - size)
    w_end = w_start + size
    h_end = h_start + size
    img = img.crop((w_start, h_start, w_end, h_end))
    return img


def random_resized_crop(img, size):
    for attempt in range(10):
        area = img.size[0] * img.size[1]
        target_area = random.uniform(0.08, 1.0) * area
        aspect_ratio = random.uniform(3. / 4, 4. / 3)

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if random.random() < 0.5:
            w, h = h, w

        if w <= img.size[0] and h <= img.size[1]:
            x1 = random.randint(0, img.size[0] - w)
            y1 = random.randint(0, img.size[1] - h)

            img = img.crop((x1, y1, x1 + w, y1 + h))
            assert (img.size == (w, h))

            return img.resize((size, size), Image.BILINEAR)

    w = min(img.size[0], img.size[1])
    i = (img.size[1] - w) // 2
    j = (img.size[0] - w) // 2
    img = img.crop((i, j, i + w, j + w))
    img = img.resize((size, size), Image.BILINEAR)
    return img


def random_crop(img, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
    aspect_ratio = math.sqrt(random.uniform(*ratio))
    w = 1. * aspect_ratio
    h = 1. / aspect_ratio

    bound = min((float(img.size[0]) / img.size[1]) / (w**2),
                (float(img.size[1]) / img.size[0]) / (h**2))
    scale_max = min(scale[1], bound)
    scale_min = min(scale[0], bound)

    target_area = img.size[0] * img.size[1] * random.uniform(scale_min,
                                                             scale_max)
    target_size = math.sqrt(target_area)
    w = int(target_size * w)
    h = int(target_size * h)

    i = random.randint(0, img.size[0] - w)
    j = random.randint(0, img.size[1] - h)

    img = img.crop((i, j, i + w, j + h))
    img = img.resize((size, size), Image.BILINEAR)
    return img


def rotate_image(img):
    angle = random.randint(-10, 10)
    img = img.rotate(angle)
    return img


def distort_color(img):
    def random_brightness(img, lower=0.8, upper=1.2):
        e = random.uniform(lower, upper)
        return ImageEnhance.Brightness(img).enhance(e)

    def random_contrast(img, lower=0.8, upper=1.2):
        e = random.uniform(lower, upper)
        return ImageEnhance.Contrast(img).enhance(e)

    def random_color(img, lower=0.8, upper=1.2):
        e = random.uniform(lower, upper)
        return ImageEnhance.Color(img).enhance(e)

    ops = [random_brightness, random_contrast, random_color]
    random.shuffle(ops)

    img = ops[0](img)
    img = ops[1](img)
    img = ops[2](img)

    return img


def process_image_imagepath(sample, class_dim, color_jitter, rotate,
                            rand_mirror, normalize):
    imgpath = sample[0]
    img = Image.open(imgpath)

    if rotate:
        img = rotate_image(img)
        img = random_resized_crop(img, DATA_DIM)

    if color_jitter:
        img = distort_color(img)

    if rand_mirror:
        if random.randint(0, 1) == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = np.array(img).astype('float32').transpose((2, 0, 1))

    if normalize:
        img -= img_mean
        img /= img_std

    assert sample[1] < class_dim, \
        "label of train dataset should be less than the class_dim."

    return img, sample[1]


def arc_iterator(data,
                 class_dim,
                 data_dir,
                 shuffle=False,
                 color_jitter=False,
                 rotate=False,
                 rand_mirror=False,
                 normalize=False):
    def reader():
        if shuffle:
            random.shuffle(data)
        for j in range(len(data)):
            path, label = data[j]
            path = os.path.join(data_dir, path)
            yield path, label

    mapper = functools.partial(
        process_image_imagepath,
        class_dim=class_dim,
        color_jitter=color_jitter,
        rotate=rotate,
        rand_mirror=rand_mirror,
        normalize=normalize)
    return paddle.reader.xmap_readers(mapper, reader, THREAD, BUF_SIZE)


def load_bin(path, image_size):
    if six.PY2:
        bins, issame_list = pickle.load(open(path, 'rb'))
    else:
        bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    data_list = []
    for flip in [0, 1]:
        data = np.empty(
            (len(issame_list) * 2, 3, image_size[0], image_size[1]))
        data_list.append(data)
    for i in range(len(issame_list) * 2):
        _bin = bins[i]
        if six.PY2:
            if not isinstance(_bin, six.string_types):
                _bin = _bin.tostring()
            img_ori = Image.open(StringIO(_bin))
        else:
            img_ori = Image.open(BytesIO(_bin))
        for flip in [0, 1]:
            img = img_ori.copy()
            if flip == 1:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = np.array(img).astype('float32').transpose((2, 0, 1))
            img -= img_mean
            img /= img_std
            data_list[flip][i][:] = img
        if i % 1000 == 0:
            print('loading bin', i)
    print(data_list[0].shape)
    return data_list, issame_list


def arc_train(data_dir, class_dim):
    train_image_list = get_train_image_list(data_dir)
    return arc_iterator(
        train_image_list,
        class_dim=class_dim,
        data_dir=data_dir,
        shuffle=True,
        color_jitter=False,
        rotate=False,
        rand_mirror=True,
        normalize=True)


def test(data_dir, datasets):
    test_list = []
    test_name_list = []
    for name in datasets.split(','):
        path = os.path.join(data_dir, name + ".bin")
        if os.path.exists(path):
            data_set = load_bin(path, (DATA_DIM, DATA_DIM))
            test_list.append(data_set)
            test_name_list.append(name)
            print('test', name)
    return test_list, test_name_list


class TrainDataset(paddle.io.Dataset):
    def __init__(self,
                 data_dir,
                 class_dim,
                 color_jitter=False,
                 rotate=False,
                 rand_mirror=False,
                 normalize=False):
        self.data_dir = data_dir
        self.class_dim = class_dim
        self.color_jitter = color_jitter
        self.rotate = rotate
        self.rand_mirror = rand_mirror
        self.normalize = normalize
        self.sample_list = get_train_image_list(data_dir)

    def __getitem__(self, idx):
        img_path, label = self.sample_list[idx]
        img_path = os.path.join(self.data_dir, img_path)
        img = Image.open(img_path)
        if self.rotate:
            img = rotate_image(img)
            img = random_resized_crop(img, DATA_DIM)

        if self.color_jitter:
            img = distort_color(img)

        if self.rand_mirror:
            if random.randint(0, 1) == 1:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

        if img.mode != 'RGB':
            img = img.convert('RGB')

        img = np.array(img).astype('float32').transpose((2, 0, 1))

        if self.normalize:
            img -= img_mean
            img /= img_std

        assert label < self.class_dim, \
            "label of train dataset should be less than the class_dim."

        return img, label

    def __len__(self):
        return len(self.sample_list)


class TestDataset(paddle.io.Dataset):
    def __init__(self, data_dir, datasets):
        self.data_dir = data_dir
        self.datasets = datasets.split(',')
        self.sample_list = []

    def __getitem__(self, idx):
        img_path, label = self.sample_list[idx]
        return img, label

    def __len__(self):
        return len(self.sample_list)
