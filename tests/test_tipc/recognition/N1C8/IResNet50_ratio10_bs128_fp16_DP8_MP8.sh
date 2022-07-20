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

model_item=IResNet50_ratio10
fp_item=fp16
bs_item=128
run_mode=DP8_MP8
device_num=N1C8
yaml_path=./plsc/configs/FaceRecognition/IResNet50_MS1MV3_ArcFace_1.0_1n8c_dp_mp_fp16o2.yaml
epochs=1

# bash ./tests/test_tipc/recognition/benchmark_common/prepare.sh

python3 -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
unset http_proxy https_proxy
python3 -m pip install -r requirements.txt #-i https://pypi.tuna.tsinghua.edu.cn/simple
# download dummy dataset
wget https://plsc.bj.bcebos.com/dataset/MS1M_v2.tgz
# unzip
mkdir -p ./dataset/
tar -xzf MS1M_v2.tgz -C ./dataset/
# convert test bin file to images and label.txt
python plsc/data/dataset/tools/lfw_style_bin_dataset_converter.py --bin_path ./dataset/MS1M_v2/agedb_30.bin --out_dir ./dataset/MS1M_v2/agedb_30/ --flip_test
# In order not to modify the configuration file, temporarily rename the dataset name.
mv ./dataset/MS1M_v2 ./dataset/MS1M_v3

# run
bash ./tests/test_tipc/recognition/benchmark_common/run_benchmark.sh ${model_item} ${fp_item} ${bs_item} ${run_mode} ${device_num} ${yaml_path} ${epochs} 2>&1;
