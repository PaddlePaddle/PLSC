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

export plsc_path=/paddle/PLSC
export log_path=/paddle/log_plsc
plsc_gpu_model_list=( \
    IResNet50_MS1MV3_ArcFace_pfc10_1n8c_dp_mp_fp16o1 \
)

function IResNet50_MS1MV3_ArcFace_pfc10_1n8c_dp_mp_fp16o1() {
    cd ${plsc_path}
    rm -rf log
    bash ./tests/CI/recognition/face/IResNet50_MS1MV3_ArcFace_pfc10_1n8c_dp_mp_fp16o1.sh
    check_result $FUNCNAME
    loss=`tail log/workerlog.0 | grep "199/5059" | cut -d " " -f12 `
    check_diff 45.00194 ${loss%?} ${FUNCNAME}_loss
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
