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

from plsc import Entry

if __name__ == "__main__":
    ins = Entry()
    #ins.set_dataset_dir('/path/to/your/data/folder/')
    ins.set_step_boundaries((100000, 160000, 220000))
    ins.set_loss_type('arcface')
    ins.set_model_parallel(True)
    ins.set_sample_ratio(0.1)
    #ins.set_mixed_precision(True)
    #ins.set_train_steps(180000)
    ins.set_train_epochs(50)
    ins.set_test_period(2000)
    ins.set_calc_acc(True)
    ins.set_model_save_dir('./saved_model')
    ins.train()
