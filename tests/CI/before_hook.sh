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
export data_path=/plsc_data
export log_path=/paddle/log_plsc
mkdir -p ${log_path}

function before_hook() {
    echo "=============================paddle commit============================="
    python -c "import paddle;print(paddle.__git_commit__)"

    # install requirements
    cd /paddle/PLSC/
    echo ---------- install plsc ----------
    export http_proxy=${proxy};
    export https_proxy=${proxy};
    python -m pip install -r requirements.txt --force-reinstall
    python -m pip install protobuf==3.20 --force-reinstall
    python setup.py develop

    
    echo ---------- ln plsc_data start ----------
    cd ${plsc_path}
    rm -rf dataset && mkdir dataset
    ln -s ${data_path}/MS1M_v3 ./dataset/MS1M_v3
    ln -s ${data_path}/ILSVRC2012 ./dataset/ILSVRC2012
    echo ---------- ln plsc_data done ---------- 
}

main() {
    before_hook
}

main$@
