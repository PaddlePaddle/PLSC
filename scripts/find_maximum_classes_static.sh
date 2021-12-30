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

export FLAGS_allocator_strategy=naive_best_fit
export FLAGS_fraction_of_gpu_memory_to_use=0.9999
python -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 tools/train.py \
    --config_file configs/ms1mv3_r50.py \
    --is_static True \
    --backbone FresResNet50 \
    --classifier LargeScaleClassifier \
    --embedding_size 512 \
    --sample_ratio 0.1 \
    --loss ArcFace \
    --batch_size 64 \
    --num_classes 60000000 \
    --use_synthetic_dataset True \
    --do_validation_while_train False \
    --log_interval_step 1 \
    --fp16 True \
    --lsc_init_from_numpy False \
    --output fp16_arcface_static_0.1_maximum_classes
