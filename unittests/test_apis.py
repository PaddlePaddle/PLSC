# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function
import numpy as np
import os

import paddle.fluid as fluid
import plsc
from plsc import Entry
from plsc.version import plsc_version
import plsc.config as config
from plsc.models import ResNet50


if __name__ == "__main__":
    passed_apis = 0
    failed_apis = 0

    ins = Entry()

    for key in config.config:
        assert ins.config[key] == config.config[key]

    assert plsc_version == plsc.__version__
    assert ins.num_trainers == 1
    assert ins.trainer_id == 0

    # Test apis
    # 1. set_val_targets
    ins.set_val_targets("lfw,cfp_ff,ctp_fp,custom")
    if ins.val_targets == "lfw,cfp_ff,ctp_fp,custom":
        print("set_val_targets passed")
        passed_apis += 1
    else:
        print("set_val_targets failed")
        failed_apis += 1

    # 2. set_train_batch_size
    batch_size = np.random.randint(1, 100)
    ins.set_train_batch_size(batch_size)
    if (ins.train_batch_size == batch_size and 
            ins.global_train_batch_size == batch_size * ins.num_trainers):
        print("set_train_batch_size passed")
        passed_apis += 1
    else:
        print("set_train_batch_size failed")
        failed_apis += 1

    # 3. set_test_batch_size
    batch_size = np.random.randint(1, 80)
    ins.set_test_batch_size(batch_size)
    if (ins.test_batch_size == batch_size and 
            ins.global_test_batch_size == batch_size * ins.num_trainers):
        print("set_test_batch_size passed")
        passed_apis += 1
    else:
        print("set_test_batch_size failed")
        failed_apis += 1

    # 4. set_hdfs_info
    fs_name = "afs://xingtian.afs.baidu.com:9902"
    fs_ugi = "paddle,paddle"
    directory = "remote"
    ins.set_hdfs_info(fs_name, fs_ugi, directory)
    if (ins.fs_name == fs_name and ins.fs_ugi == fs_ugi and
            ins.fs_dir_for_save == directory):
        print("set_hdfs_info passed")
        passed_apis += 1
    else:
        print("set_hdfs_info failed")
        failed_apis += 1

    # 5. set_model_save_dir
    model_save_dir = "model_dir_" + str(np.random.randint(1, 100))
    ins.set_model_save_dir(model_save_dir)
    model_save_dir = os.path.abspath(model_save_dir)
    if ins.model_save_dir == model_save_dir:
        print("set_model_save_dir passed")
        passed_apis += 1
    else:
        print("set_model_save_dir failed")
        failed_apis += 1

    # 6. set_calc_acc
    calc = np.random.randint(0, 2)
    calc = True if calc > 0 else False
    ins.set_calc_acc(calc=calc)
    if ins.calc_train_acc == calc:
        print("set_calc_acc passed")
        passed_apis += 1
    else:
        print("set_calc_acc failed")
        failed_apis += 1

    # 7. set_dataset_dir
    dataset_dir = "dataset_dir_" + str(np.random.randint(1, 100))
    ins.set_dataset_dir(dataset_dir)
    dataset_dir = os.path.abspath(dataset_dir)
    if ins.dataset_dir == dataset_dir:
        print("set_dataset_dir passed")
        passed_apis += 1
    else:
        print("set_dataset_dir failed")
        failed_apis += 1

    # 8. set_train_image_num
    image_num = np.random.randint(100, 1000)
    ins.set_train_image_num(image_num)
    if ins.train_image_num == image_num:
        print("set_train_image_num passed")
        passed_apis += 1
    else:
        print("set_train_image_num failed")
        failed_apis += 1

    # 9. set_class_num
    class_num = np.random.randint(1000, 10000)
    ins.set_class_num(class_num)
    if ins.num_classes == class_num:
        print("set_class_num passed")
        passed_apis += 1
    else:
        print("set_class_num failed")
        failed_apis += 1

    # 10. set_emb_size
    emb_size = np.random.randint(1000, 1025)
    ins.set_emb_size(emb_size)
    if ins.emb_dim == emb_size:
        print("set_emb_size passed")
        passed_apis += 1
    else:
        print("set_emb_size failed")
        failed_apis += 1

    # 11. set_model
    model = ResNet50()
    ins.set_model(model)
    if ins.model != model:
        print("set_model failed")
        failed_apis += 1
    model = "resnet50"
    try:
        ins.set_model(model)
        print("set_model failed")
        failed_apis += 1
    except ValueError:
        print("set_model passed")
        passed_apis += 1

    # 12. set_train_epochs
    num_epochs = np.random.randint(200, 300)
    ins.set_train_epochs(num_epochs)
    if ins.train_epochs == num_epochs:
        print("set_train_epochs passed")
        passed_apis += 1
    else:
        print("set_train_epochs failed")
        failed_apis += 1

    # 13. set_checkpoint_dir
    checkpoint_dir = "checkpoint_dir_" + str(np.random.randint(1, 100))
    ins.set_checkpoint_dir(checkpoint_dir)
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    if ins.checkpoint_dir == checkpoint_dir:
        print("set_checkpoint_dir passed")
        passed_apis += 1
    else:
        print("set_checkpoint_dir failed")
        failed_apis += 1

    # 14. set_warmup_epochs
    warmup_epochs = np.random.randint(10, 100)
    ins.set_warmup_epochs(warmup_epochs)
    if ins.warmup_epochs == warmup_epochs:
        print("set_warmup_epochs passed")
        passed_apis += 1
    else:
        print("set_warmup_epochs failed")
        failed_apis += 1

    # 15. set_loss_type
    loss_type = "arcface"
    ins.set_loss_type(loss_type)
    if ins.loss_type != loss_type:
        print("set_loss_type failed")
        failed_apis += 1
    loss_type = "not_supported"
    try:
        ins.set_loss_type(loss_type)
        print("set_loss_type failed")
        failed_apis += 1
    except ValueError:
        print("set_loss_type passed")
        passed_apis += 1

    # 16. set_image_shape
    shape = (3, 224, 224)
    ins.set_image_shape(shape)
    if ins.image_shape != shape:
        print("set_image_shape failed")
        failed_apis += 1
    shape = 3
    try:
        ins.set_image_shape(shape)
        print("set_image_shape failed")
        failed_apis += 1
    except ValueError:
        print("set_image_shape passed")
        passed_apis += 1


    # 17. set_optimizer
    optimizer = fluid.optimizer.Momentum(learning_rate=0.1, momentum=0.9)
    ins.set_optimizer(optimizer)
    if ins.optimizer != optimizer:
        print("set_optimizer failed")
        failed_apis += 1
    optimizer = "opt"
    try:
        ins.set_optimizer(optimizer)
        print("set_optimizer failed")
        failed_apis += 1
    except ValueError:
        print("set_optimizer passed")
        passed_apis += 1

    # 18. set_with_test
    with_test = np.random.randint(0, 2)
    with_test = True if with_test > 0 else False
    ins.set_with_test(with_test)
    if ins.with_test == with_test:
        print("set_with_test passed")
        passed_apis += 1
    else:
        print("set_with_test failed")
        failed_apis += 1

    ignore_keys = ['with_test', 'scale', 'margin', 'lr', 'lr_steps',
                  'model_name']
    for key in config.config:
        assert hasattr(ins, key)
        if key in ignore_keys:
            continue
        if getattr(ins, key) == config.config[key]:
            raise ValueError("{}:{}".format(key, config.config[key]))

    print("Done!")
    print("Passed APIs: ", passed_apis)
    print("Failed APIs: ", failed_apis)
