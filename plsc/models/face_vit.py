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

# Code based on: 
# https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/backbones/vit.py

from collections.abc import Callable
import collections

import os
import math
import warnings
import numpy as np
import paddle
import paddle.nn as nn

from plsc.utils import logger
from plsc.nn import PartialFC
from plsc.nn import init
from plsc.models.base_model import Model

from .vision_transformer import to_2tuple, Mlp, PatchEmbed, DropPath, Attention, Block

__all__ = [
    'FaceViT_tiny_patch9_112', 'FaceViT_small_patch9_112',
    'FaceViT_base_patch9_112', 'FaceViT_large_patch9_112', 'FaceViT'
]


class FaceViT(Model):
    """ Vision Transformer with support for patch input
    """

    def __init__(self,
                 img_size=112,
                 patch_size=16,
                 in_chans=3,
                 num_features=512,
                 class_num=93431,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer='nn.LayerNorm',
                 epsilon=1e-5,
                 mask_ratio=0.1,
                 pfc_config={"model_parallel": True,
                             "sample_ratio": 1.0},
                 **kwargs):
        super().__init__()
        self.class_num = class_num
        self.mask_ratio = mask_ratio

        self.num_features = num_features
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        self.pos_embed = self.create_parameter(
            shape=(1, num_patches, embed_dim),
            default_initializer=paddle.nn.initializer.Constant(value=0.))

        self.mask_token = self.create_parameter(
            shape=(1, 1, embed_dim),
            default_initializer=paddle.nn.initializer.Constant(value=0.))

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = np.linspace(0, drop_path_rate, depth)

        self.blocks = nn.LayerList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=nn.ReLU6,
                epsilon=epsilon) for i in range(depth)
        ])

        self.norm = eval(norm_layer)(embed_dim, epsilon=epsilon)

        # features
        self.feature = nn.Sequential(
            nn.Linear(
                in_features=embed_dim * num_patches,
                out_features=embed_dim,
                bias_attr=False),
            nn.BatchNorm1D(
                num_features=embed_dim, epsilon=2e-5),
            nn.Linear(
                in_features=embed_dim,
                out_features=num_features,
                bias_attr=False),
            nn.BatchNorm1D(
                num_features=num_features, epsilon=2e-5))

        pfc_config.update({
            'num_classes': class_num,
            'embedding_size': num_features,
            'name': 'partialfc'
        })
        self.head = PartialFC(**pfc_config)

        init.trunc_normal_(self.mask_token)
        init.trunc_normal_(self.pos_embed)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            init.zeros_(m.bias)
            init.ones_(m.weight)

    def random_masking(self, x, mask_ratio=0.1):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        # noise in [0, 1]
        if x.dtype == paddle.float16:
            noise = paddle.rand((N, L), dtype=paddle.float32).astype(x.dtype)
        else:
            noise = paddle.rand((N, L), dtype=x.dtype)

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = paddle.argsort(noise, axis=1)
        ids_restore = paddle.argsort(ids_shuffle, axis=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        x_masked = paddle.take_along_axis(
            x, ids_keep.unsqueeze(-1).tile([1, 1, D]), axis=1)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = paddle.ones([N, L])
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = paddle.take_along_axis(mask, ids_restore, axis=1)

        return x_masked, mask, ids_restore

    def forward_features(self, x):
        B = paddle.shape(x)[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        if self.training and self.mask_ratio > 0:
            x, _, ids_restore = self.random_masking(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.training and self.mask_ratio > 0:
            mask_tokens = self.mask_token.tile(
                [x.shape[0], ids_restore.shape[1] - x.shape[1], 1])
            x_ = paddle.concat([x, mask_tokens], axis=1)  # no cls token
            x_ = paddle.take_along_axis(
                x_, ids_restore.unsqueeze(-1).tile([1, 1, x.shape[2]]),
                axis=1)  # unshuffle
            x = x_
        return paddle.reshape(x, [B, self.num_patches * self.embed_dim])

    def forward(self, inputs):
        if isinstance(inputs, dict):
            x = inputs['data']
        else:
            x = inputs
        x.stop_gradient = True
        x = self.forward_features(x)
        y = self.feature(x)

        if not self.training:
            # return embedding feature
            if isinstance(inputs, dict):
                res = {'logits': y}
                if 'targets' in inputs:
                    res['targets'] = inputs['targets']
            else:
                res = y
            return res

        assert isinstance(inputs, dict) and 'targets' in inputs
        y, targets = self.head(y, inputs['targets'])

        return {'logits': y, 'targets': targets}

    def load_pretrained(self, path, rank=0, finetune=False):
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
            return

        return

    def save(self, path, local_rank=0, rank=0):
        dist_state_dict = collections.OrderedDict()
        state_dict = self.state_dict()
        for name, param in list(state_dict.items()):
            if param.is_distributed:
                dist_state_dict[name] = state_dict.pop(name)

        if local_rank == 0:
            paddle.save(state_dict, path + ".pdparams")

        if len(dist_state_dict) > 0:
            paddle.save(dist_state_dict,
                        path + "_rank{}.pdparams".format(rank))


def FaceViT_tiny_patch9_112(**kwargs):
    model = FaceViT(
        img_size=112,
        patch_size=9,
        embed_dim=256,
        depth=12,
        num_heads=8,
        mlp_ratio=4,
        **kwargs)
    return model


def FaceViT_small_patch9_112(**kwargs):
    model = FaceViT(
        img_size=112,
        patch_size=9,
        embed_dim=512,
        depth=12,
        num_heads=8,
        mlp_ratio=4,
        **kwargs)
    return model


def FaceViT_base_patch9_112(**kwargs):
    model = FaceViT(
        img_size=112,
        patch_size=9,
        embed_dim=512,
        depth=24,
        num_heads=8,
        mlp_ratio=4,
        **kwargs)
    return model


def FaceViT_large_patch9_112(**kwargs):
    model = FaceViT(
        img_size=112,
        patch_size=9,
        embed_dim=768,
        depth=24,
        num_heads=8,
        mlp_ratio=4,
        **kwargs)
    return model
