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

python -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 tools/train.py \
    --config_file configs/ms1mv2_mobileface.py \
    --is_static False \
    --backbone MobileFaceNet_128 \
    --classifier LargeScaleClassifier \
    --embedding_size 128 \
    --model_parallel True \
    --dropout 0.0 \
    --sample_ratio 1.0 \
    --loss ArcFace \
    --batch_size 128 \
    --dataset MS1M_v2 \
    --num_classes 85742 \
    --data_dir MS1M_v2/ \
    --label_file MS1M_v2/label.txt \
    --is_bin False \
    --log_interval_step 100 \
    --validation_interval_step 2000 \
    --fp16 False \
    --use_dynamic_loss_scaling True \
    --train_unit 'epoch' \
    --output ./MS1M_v2_arcface_MobileFaceNet_128
