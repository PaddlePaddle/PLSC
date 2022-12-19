#!/usr/bin/env bash

# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

# for single card training
# CUDA_VISIBLE_DEVICES=0
# python tools/train.py -c ./plsc/configs/FaceRecognition/IResNet50_MS1MV3_ArcFace_0.1_1n1c_dp_fp16o2.yaml

# for multi-node and multi-cards training
# export PADDLE_NNODES=2
# export PADDLE_MASTER="192.168.210.1:12538"
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# for single-node and multi-cards training
export PADDLE_NNODES=1
export PADDLE_MASTER="127.0.0.1:12538"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch \
    --nnodes=$PADDLE_NNODES \
    --master=$PADDLE_MASTER \
    --devices=$CUDA_VISIBLE_DEVICES \
    plsc-train \
    -c task/recognition/face/configs/IResNet50_MS1MV3_ArcFace_pfc10_1n8c_dp_mp_fp16o1.yaml \
    -o DataLoader.Train.dataset.image_root=./dataset/MS1M_v3_One_Sample \
    -o DataLoader.Train.dataset.cls_label_path=./dataset/MS1M_v3_One_Sample/label.txt \
    -o DataLoader.Eval.dataset.image_root=./dataset/MS1M_v3_One_Sample/agedb_30 \
    -o DataLoader.Eval.dataset.cls_label_path=./dataset/MS1M_v3_One_Sample/agedb_30/label.txt
