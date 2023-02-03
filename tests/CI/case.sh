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

#!/usr/bin/env bash
set -e

export plsc_path=/paddle/PLSC/tests/CI
export log_path=/paddle/log_plsc
plsc_gpu_model_list=( \
    IResNet50_MS1MV3_ArcFace_pfc10_1n8c_dp_mp_fp16o1 \
    FaceViT_tiny_patch9_112_WebFace42M_CosFace_pfc10_droppath005_mask0_1n8c_dp_mp_fp16o1 \
    FaceViT_tiny_patch9_112_WebFace42M_CosFace_pfc02_droppath005_mask0_1n8c_dp_mp_fp16o1 \
    FaceViT_base_patch9_112_WebFace42M_CosFace_pfc03_droppath005_mask005_1n8c_dp_mp_fp16o1 \
    IResNet100_WebFace42M_CosFace_pfc02_1n8c_dp_mp_fp16o1 \
    IResNet50_MS1MV3_ArcFace_pfc01_1n1c_fp16o1 \
    IResNet50_MS1MV3_ArcFace_pfc01_1n8c_dp8_fp16o1 \
    ViT_base_patch16_224_in1k_1n8c_dp_fp16o2 \
    ViT_base_patch16_384_ft_in1k_1n8c_dp_fp16o2 \
    DeiT_base_patch16_224_in1k_1n8c_dp_fp32 \
    DeiT_base_patch16_224_in1k_1n8c_dp_fp16o2 \
    cait_s24_224_in1k_1n8c_dp_fp16o2 \
    swin_base_patch4_window7_224_fp16o2 \
)

###### Face ######
function IResNet50_MS1MV3_ArcFace_pfc10_1n8c_dp_mp_fp16o1() {
    cd ${plsc_path}
    rm -rf log
    bash ./recognition/face/IResNet50_MS1MV3_ArcFace_pfc10_1n8c_dp_mp_fp16o1.sh
    check_result $FUNCNAME
    loss=`tail log/workerlog.0 | grep "199/5059" | cut -d " " -f12 `
    check_diff 44.58809 ${loss%?} ${FUNCNAME}_loss
}

function IResNet100_WebFace42M_CosFace_pfc02_1n8c_dp_mp_fp16o1() {
    cd ${plsc_path}
    rm -rf log
    bash ./recognition/face/IResNet100_WebFace42M_CosFace_pfc02_1n8c_dp_mp_fp16o1.sh
    check_result $FUNCNAME
    loss=`tail log/workerlog.0 | grep "199/40465" | cut -d " " -f12 `
    check_diff 40.74793 ${loss%?} ${FUNCNAME}_loss
}

function FaceViT_tiny_patch9_112_WebFace42M_CosFace_pfc10_droppath005_mask0_1n8c_dp_mp_fp16o1() {
    cd ${plsc_path}
    rm -rf log
    bash ./recognition/face/FaceViT_tiny_patch9_112_WebFace42M_CosFace_pfc10_droppath005_mask0_1n8c_dp_mp_fp16o1.sh
    check_result $FUNCNAME
    loss=`tail log/workerlog.0 | grep "199/2530" | cut -d " " -f12 `
    check_diff 38.36615 ${loss%?} ${FUNCNAME}_loss
}

function FaceViT_tiny_patch9_112_WebFace42M_CosFace_pfc02_droppath005_mask0_1n8c_dp_mp_fp16o1() {
    cd ${plsc_path}
    rm -rf log
    bash ./recognition/face/FaceViT_tiny_patch9_112_WebFace42M_CosFace_pfc02_droppath005_mask0_1n8c_dp_mp_fp16o1.sh
    check_result $FUNCNAME
    loss=`tail log/workerlog.0 | grep "199/2530" | cut -d " " -f12 `
    check_diff 37.72491 ${loss%?} ${FUNCNAME}_loss
}

function FaceViT_base_patch9_112_WebFace42M_CosFace_pfc03_droppath005_mask005_1n8c_dp_mp_fp16o1() {
    cd ${plsc_path}
    rm -rf log
    bash ./recognition/face/FaceViT_base_patch9_112_WebFace42M_CosFace_pfc03_droppath005_mask005_1n8c_dp_mp_fp16o1.sh
    check_result $FUNCNAME
    loss=`tail log/workerlog.0 | grep "199/5059" | cut -d " " -f12 `
    check_diff 38.86674 ${loss%?} ${FUNCNAME}_loss
}

function IResNet50_MS1MV3_ArcFace_pfc01_1n1c_fp16o1() {
    cd ${plsc_path}
    rm -rf log
    bash ./recognition/face/IResNet50_MS1MV3_ArcFace_pfc01_1n1c_fp16o1.sh
    check_result $FUNCNAME
    loss=`tail log/workerlog.0 | grep "199/40465" | cut -d " " -f12 `
    check_diff 46.27080 ${loss%?} ${FUNCNAME}_loss
}

function IResNet50_MS1MV3_ArcFace_pfc01_1n8c_dp8_fp16o1() {
    cd ${plsc_path}
    rm -rf log
    bash ./recognition/face/IResNet50_MS1MV3_ArcFace_pfc01_1n8c_dp8_fp16o1.sh
    check_result $FUNCNAME
    loss=`tail log/workerlog.0 | grep "199/5059" | cut -d " " -f12 `
    check_diff 41.70836 ${loss%?} ${FUNCNAME}_loss
}

#function MobileFaceNet_WebFace42M_CosFace_pfc02_1n8c_dp_mp_fp16o1() {
#    cd ${plsc_path}
#    rm -rf log
#    bash ./recognition/face/MobileFaceNet_WebFace42M_CosFace_pfc02_1n8c_dp_mp_fp16o1.sh
#    check_result $FUNCNAME
#    loss=`tail log/workerlog.0 | grep "199/20233" | cut -d " " -f12 `
#    check_diff 42.17661 ${loss%?} ${FUNCNAME}_loss
#}

###### ViT ######
function ViT_base_patch16_224_in1k_1n8c_dp_fp16o2() {
    cd ${plsc_path}
    rm -rf log
    bash ./classification/vit/ViT_base_patch16_224_in1k_1n8c_dp_fp16o2.sh
    check_result $FUNCNAME
    loss=`tail log/workerlog.0 | grep "49/313" | cut -d " " -f18 `
    check_diff 10.90618 ${loss%?} ${FUNCNAME}_loss
}

function ViT_base_patch16_384_ft_in1k_1n8c_dp_fp16o2() {
    cd ${plsc_path}
    rm -rf log
    bash ./classification/vit/ViT_base_patch16_384_ft_in1k_1n8c_dp_fp16o2.sh
    check_result $FUNCNAME
    loss=`tail log/workerlog.0 | grep "49/2502" | cut -d " " -f18 `
    check_diff 6.90645 ${loss%?} ${FUNCNAME}_loss
}


###### DeiT ######
function DeiT_base_patch16_224_in1k_1n8c_dp_fp32() {
    cd ${plsc_path}
    rm -rf log
    bash ./classification/deit/DeiT_base_patch16_224_in1k_1n8c_dp_fp32.sh
    check_result $FUNCNAME
    loss=`tail log/workerlog.0 | grep "49/1251" | cut -d " " -f12 `
    check_diff 6.90764 ${loss%?} ${FUNCNAME}_loss
}


function DeiT_base_patch16_224_in1k_1n8c_dp_fp16o2() {
    cd ${plsc_path}
    rm -rf log
    bash ./classification/deit/DeiT_base_patch16_224_in1k_1n8c_dp_fp16o2.sh
    check_result $FUNCNAME
    loss=`tail log/workerlog.0 | grep "49/1251" | cut -d " " -f12 `
    check_diff 6.91250 ${loss%?} ${FUNCNAME}_loss
}


###### CaiT ######
function cait_s24_224_in1k_1n8c_dp_fp16o2() {
    cd ${plsc_path}
    rm -rf log
    bash ./classification/cait/cait_s24_224_in1k_1n8c_dp_fp16o2.sh
    check_result $FUNCNAME
    loss=`tail log/workerlog.0 | grep "49/1251" | cut -d " " -f12 `
    check_diff 6.98169 ${loss%?} ${FUNCNAME}_loss
}


###### Swin ######
function swin_base_patch4_window7_224_fp16o2() {
    cd ${plsc_path}
    rm -rf log
    bash ./classification/swin/swin_base_patch4_window7_224_fp16o2.sh
    check_result $FUNCNAME
    loss=`tail log/workerlog.0 | grep "49/1252" | cut -d " " -f12 `
    check_diff 6.98169 ${loss%?} ${FUNCNAME}_loss
}


function check_result() {
    if [ $? -ne 0 ];then
      echo -e "\033 $1 model runs failed! \033" | tee -a $log_path/result.log
    else
      echo -e "\033 $1 model runs successfully! \033" | tee -a $log_path/result.log
    fi
}

function check_diff() {
    echo "base:$1 test:$2"
    if [ $1 != $2 ];then
      echo -e "\033 $3 model_diff runs failed! \033" | tee -a $log_path/result.log
      exit -1
    else
      echo -e "\033 $3 model_diff runs successfully! \033" | tee -a $log_path/result.log
    fi
}

function run_gpu_models(){
    cd
      for model in ${plsc_gpu_model_list[@]}
      do
        echo "=========== ${model} run begin ==========="
        $model
        echo "=========== ${model} run  end ==========="
      done
}

main() {
    run_gpu_models
}

main$@
