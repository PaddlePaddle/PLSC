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

import argparse
import pickle
import os
import six
import numpy as np
from PIL import Image
from io import BytesIO


def main(args):
    bins, issame_list = pickle.load(
        open(args.bin_path, 'rb'), encoding='bytes')

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    label_path = os.path.join(args.out_dir, 'label.txt')
    image_dir = os.path.join(args.out_dir, 'images')

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    fp = open(label_path, 'w')

    for i in range(len(issame_list)):
        _bin = bins[2 * i]
        img1 = Image.open(BytesIO(_bin))
        if img1.mode != 'RGB':
            img1 = img1.convert('RGB')
        img1_path = os.path.join(image_dir, '%08d_0.png' % i)
        img1.save(img1_path, format="png")

        _bin = bins[2 * i + 1]
        img2 = Image.open(BytesIO(_bin))
        if img2.mode != 'RGB':
            img2 = img2.convert('RGB')
        img2_path = os.path.join(image_dir, '%08d_1.png' % i)
        img2.save(img2_path, format="png")

        fp.write('images/%08d_0.png\timages/%08d_1.png\t%d\n' %
                 (i, i, issame_list[i]))

    print('convert {} pair images.'.format(len(issame_list)))

    if args.flip_test:
        for i in range(len(issame_list)):
            _bin = bins[2 * i]
            img1 = Image.open(BytesIO(_bin))
            if img1.mode != 'RGB':
                img1 = img1.convert('RGB')
            img1_path = os.path.join(image_dir, '%08d_0_hflip.png' % i)
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
            img1.save(img1_path, format="png")

            _bin = bins[2 * i + 1]
            img2 = Image.open(BytesIO(_bin))
            if img2.mode != 'RGB':
                img2 = img2.convert('RGB')
            img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
            img2_path = os.path.join(image_dir, '%08d_1_hflip.png' % i)
            img2.save(img2_path, format="png")

            fp.write('images/%08d_0_hflip.png\timages/%08d_1_hflip.png\t%d\n' %
                     (i, i, issame_list[i]))

        print('convert {} pair horizontal flip images.'.format(
            len(issame_list)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--bin_path", type=str, help="bin file path")
    parser.add_argument("--out_dir", type=str, help="output directory")
    parser.add_argument(
        "--flip_test",
        action='store_true',
        help="add flip augmentation sample")
    args = parser.parse_args()
    main(args)
