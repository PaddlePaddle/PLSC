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
from __future__ import division
import os
import sys
import time
import argparse
import numpy as np
import math
import pickle
import subprocess
import shutil
import logging
import tempfile

import paddle
import paddle.fluid as fluid
import sklearn
from . import config
from .models import resnet
from .models import base_model
from .models.dist_algo import DistributedClassificationOptimizer
from .utils.learning_rate import lr_warmup
from .utils.verification import evaluate
from .utils import jpeg_reader as reader
from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.transpiler.details.program_utils import program_to_code
import paddle.fluid.transpiler.distribute_transpiler as dist_transpiler
from paddle.fluid.optimizer import Optimizer


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%d %b %Y %H:%M:%S')
logger = logging.getLogger(__name__)
        

class Entry(object):
    """
    The class to encapsulate all operations.
    """

    def _check(self):
        """
        Check the validation of parameters.
        """
        assert os.getenv("PADDLE_TRAINERS_NUM") is not None, \
            "Please start script using paddle.distributed.launch module."

        supported_types = ["softmax", "arcface",
                           "dist_softmax", "dist_arcface"]
        assert self.loss_type in supported_types, \
            "All supported types are {}, but given {}.".format(
                supported_types, self.loss_type)

        if self.loss_type in ["dist_softmax", "dist_arcface"]:
            assert self.num_trainers > 1, \
                "At least 2 trainers are required to use distributed fc-layer."

    def __init__(self):
        self.config = config.config
        super(Entry, self).__init__()
        num_trainers = int(os.getenv("PADDLE_TRAINERS_NUM", 1))
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID", 0))

        self.trainer_id = trainer_id
        self.num_trainers = num_trainers
        self.train_batch_size = self.config.train_batch_size
        self.test_batch_size = self.config.test_batch_size
        self.global_train_batch_size = self.train_batch_size * num_trainers
        self.global_test_batch_size = self.test_batch_size * num_trainers

        self.optimizer = None
        self.model = None
        self.train_reader = None
        self.test_reader = None

        self.train_program = fluid.Program()
        self.startup_program = fluid.Program()
        self.test_program = fluid.Program()
        self.predict_program = fluid.Program()

        self.fs_name = None
        self.fs_ugi = None
        self.fs_dir = None

        self.val_targets = self.config.val_targets
        self.dataset_dir = self.config.dataset_dir
        self.num_classes = self.config.num_classes
        self.image_shape = self.config.image_shape
        self.loss_type = self.config.loss_type
        self.margin = self.config.margin
        self.scale = self.config.scale
        self.lr = self.config.lr
        self.lr_steps = self.config.lr_steps
        self.train_image_num = self.config.train_image_num
        self.model_name = self.config.model_name
        self.emb_dim = self.config.emb_dim
        self.train_epochs = self.config.train_epochs
        self.checkpoint_dir = self.config.checkpoint_dir
        self.with_test = self.config.with_test
        self.model_save_dir = self.config.model_save_dir
        self.warmup_epochs = self.config.warmup_epochs
        self.calc_train_acc = False

        if self.checkpoint_dir:
            self.checkpoint_dir = os.path.abspath(self.checkpoint_dir)
        if self.model_save_dir:
            self.model_save_dir = os.path.abspath(self.model_save_dir)
        if self.dataset_dir:
            self.dataset_dir = os.path.abspath(self.dataset_dir)

        logger.info('=' * 30)
        logger.info("Default configuration:")
        for key in self.config:
            logger.info('\t' + str(key) + ": " + str(self.config[key]))
        logger.info('trainer_id: {}, num_trainers: {}'.format(
            trainer_id, num_trainers))
        logger.info('=' * 30)

    def set_val_targets(self, targets):
        """
        Set the names of validation datasets, separated by comma.
        """
        self.val_targets = targets
        logger.info("Set val_targets to {}.".format(targets))

    def set_train_batch_size(self, batch_size):
        self.train_batch_size = batch_size
        self.global_train_batch_size = batch_size * self.num_trainers
        logger.info("Set train batch size to {}.".format(batch_size))

    def set_test_batch_size(self, batch_size):
        self.test_batch_size = batch_size
        self.global_test_batch_size = batch_size * self.num_trainers
        logger.info("Set test batch size to {}.".format(batch_size))

    def set_hdfs_info(self, fs_name, fs_ugi, directory):
        """
        Set the info to download from or upload to hdfs filesystems.
        If the information is provided, we will download pretrained 
        model from hdfs at the begining and upload pretrained models
        to hdfs at the end automatically.
        """
        self.fs_name = fs_name
        self.fs_ugi = fs_ugi
        self.fs_dir = directory
        logger.info("HDFS Info:")
        logger.info("\tfs_name: {}".format(fs_name))
        logger.info("\tfs_ugi: {}".format(fs_ugi))
        logger.info("\tremote directory: {}".format(directory))

    def set_model_save_dir(self, directory):
        """
        Set the directory to save models.
        """
        if directory:
            directory = os.path.abspath(directory)
        self.model_save_dir = directory
        logger.info("Set model_save_dir to {}.".format(directory))

    def set_calc_acc(self, calc):
        """
        Whether to calcuate acc1 and acc5 during training.
        """
        self.calc_train_acc = calc
        logger.info("Calcuating acc1 and acc5 during training: {}.".format(
            calc))

    def set_dataset_dir(self, directory):
        """
        Set the root directory for datasets.
        """
        if directory:
            directory = os.path.abspath(directory)
        self.dataset_dir = directory
        logger.info("Set dataset_dir to {}.".format(directory))

    def set_train_image_num(self, num):
        """
        Set the total number of images for train.
        """
        self.train_image_num = num
        logger.info("Set train_image_num to {}.".format(num))

    def set_class_num(self, num):
        """
        Set the number of classes.
        """
        self.num_classes = num
        logger.info("Set num_classes to {}.".format(num))

    def set_emb_size(self, size):
        """
        Set the size of the last hidding layer before the distributed fc-layer.
        """
        self.emb_size = size
        logger.info("Set emb_size to {}.".format(size))

    def set_model(self, model):
        """
        Set user-defined model to use.
        """
        self.model = model
        if not isinstance(model, base_model.BaseModel):
            raise ValueError("The parameter for set_model must be an "
                             "instance of BaseModel.")
        logger.info("Set model to {}.".format(model))

    def set_train_epochs(self, num):
        """
        Set the number of epochs to train.
        """
        self.train_epochs = num
        logger.info("Set train_epochs to {}.".format(num))

    def set_checkpoint_dir(self, directory):
        """
        Set the directory for checkpoint loaded before training/testing.
        """
        if directory:
            directory = os.path.abspath(directory)
        self.checkpoint_dir = directory
        logger.info("Set checkpoint_dir to {}.".format(directory))

    def set_warmup_epochs(self, num):
        self.warmup_epochs = num
        logger.info("Set warmup_epochs to {}.".format(num))

    def set_loss_type(self, type):
        supported_types = ["dist_softmax", "dist_arcface", "softmax", "arcface"]
        if not type in supported_types:
            raise ValueError("All supported loss types: {}".format(
                supported_types))
        self.loss_type = type
        logger.info("Set loss_type to {}.".format(type))

    def set_image_shape(self, shape):
        if not isinstance(shape, (list, tuple)):
            raise ValueError("Shape must be of type list or tuple")
        self.image_shape = shape
        logger.info("Set image_shape to {}.".format(shape))

    def set_optimizer(self, optimizer):
        if not isinstance(optimizer, Optimizer):
            raise ValueError("Optimizer must be type of Optimizer")
        self.optimizer = optimizer
        logger.info("User manually set optimizer")

    def _get_optimizer(self):
        if not self.optimizer:
            bd = [step for step in self.lr_steps]
            start_lr = self.lr
    
            global_batch_size = self.global_train_batch_size
            train_image_num = self.train_image_num
            images_per_trainer = int(math.ceil(
                train_image_num * 1.0 / self.num_trainers))
            steps_per_pass = int(math.ceil(
                images_per_trainer * 1.0 / self.train_batch_size)) 
            logger.info("Steps per epoch: %d" % steps_per_pass)
            warmup_steps = steps_per_pass * self.warmup_epochs
            batch_denom = 1024
            base_lr = start_lr * global_batch_size / batch_denom
            lr = [base_lr * (0.1 ** i) for i in range(len(bd) + 1)]
            logger.info("LR boundaries: {}".format(bd))
            logger.info("lr_step: {}".format(lr))
            if self.warmup_epochs:
                lr_val = lr_warmup(fluid.layers.piecewise_decay(boundaries=bd,
                    values=lr), warmup_steps, start_lr, base_lr)
            else:
                lr_val = fluid.layers.piecewise_decay(boundaries=bd, values=lr)
    
            optimizer = fluid.optimizer.Momentum(
                learning_rate=lr_val, momentum=0.9,
                regularization=fluid.regularizer.L2Decay(5e-4))
            self.optimizer = optimizer
    
        if self.loss_type in ["dist_softmax", "dist_arcface"]:
            self.optimizer = DistributedClassificationOptimizer(
                self.optimizer, global_batch_size)

        return self.optimizer

    def build_program(self,
                      is_train=True,
                      use_parallel_test=False):
        model_name = self.model_name
        assert not (is_train and use_parallel_test), \
            "is_train and use_parallel_test cannot be set simultaneously."

        trainer_id = self.trainer_id
        num_trainers = self.num_trainers

        image_shape = [int(m) for m in self.image_shape]
        # model definition
        model = self.model
        if model is None:
            model = resnet.__dict__[model_name](emb_dim=self.emb_dim)
        main_program = self.train_program if is_train else self.test_program
        startup_program = self.startup_program
        with fluid.program_guard(main_program, startup_program):
            with fluid.unique_name.guard():
                image = fluid.layers.data(name='image',
                    shape=image_shape, dtype='float32')
                label = fluid.layers.data(name='label',
                    shape=[1], dtype='int64')

                emb, loss, prob = model.get_output(
                        input=image,
                        label=label,
                        is_train=is_train,
                        num_classes=self.num_classes,
                        loss_type=self.loss_type,
                        margin=self.margin,
                        scale=self.scale)

                if self.loss_type in ["dist_softmax", "dist_arcface"]:
                    if self.calc_train_acc:
                        shard_prob = loss._get_info("shard_prob")

                        prob_all = fluid.layers.collective._c_allgather(shard_prob,
                            nranks=num_trainers, use_calc_stream=True)
                        prob_list = fluid.layers.split(prob_all, dim=0,
                            num_or_sections=num_trainers)
                        prob = fluid.layers.concat(prob_list, axis=1)
                        label_all = fluid.layers.collective._c_allgather(label,
                            nranks=num_trainers, use_calc_stream=True)
                        acc1 = fluid.layers.accuracy(input=prob, label=label_all, k=1)
                        acc5 = fluid.layers.accuracy(input=prob, label=label_all, k=5)
                    else:
                        acc1 = None
                        acc5 = None
                else:
                    if self.calc_train_acc:
                        acc1 = fluid.layers.accuracy(input=prob, label=label, k=1)
                        acc5 = fluid.layers.accuracy(input=prob, label=label, k=5)
                    else:
                        acc1 = None
                        acc5 = None
                optimizer = None
                if is_train:
                    # initialize optimizer
                    optimizer = self._get_optimizer()
                    dist_optimizer = self.fleet.distributed_optimizer(
                        optimizer, strategy=self.strategy)
                    dist_optimizer.minimize(loss)
                    if "dist" in self.loss_type:
                        optimizer = optimizer._optimizer
                elif use_parallel_test:
                    emb = fluid.layers.collective._c_allgather(emb,
                        nranks=num_trainers, use_calc_stream=True)
        return emb, loss, acc1, acc5, optimizer


    def get_files_from_hdfs(self, local_dir):
        cmd = "hadoop fs -D fs.default.name="
        cmd += self.fs_name + " "
        cmd += "-D hadoop.job.ugi="
        cmd += self.fs_ugi + " "
        cmd += "-get " + self.fs_dir
        cmd += " " + local_dir
        logger.info("hdfs download cmd: {}".format(cmd))
        cmd = cmd.split(' ')
        process = subprocess.Popen(cmd,
                         stdout=sys.stdout,
                         stderr=subprocess.STDOUT)
        process.wait()

    def put_files_to_hdfs(self, local_dir):
        cmd = "hadoop fs -D fs.default.name="
        cmd += self.fs_name + " "
        cmd += "-D hadoop.job.ugi="
        cmd += self.fs_ugi + " "
        cmd += "-put " + local_dir
        cmd += " " + self.fs_dir
        logger.info("hdfs upload cmd: {}".format(cmd))
        cmd = cmd.split(' ')
        process = subprocess.Popen(cmd,
                         stdout=sys.stdout,
                         stderr=subprocess.STDOUT)
        process.wait()

    def preprocess_distributed_params(self,
                                      local_dir):
        local_dir = os.path.abspath(local_dir)
        output_dir = tempfile.mkdtemp()
        cmd = sys.executable + ' -m plsc.utils.process_distfc_parameter '
        cmd += "--nranks {} ".format(self.num_trainers)
        cmd += "--num_classes {} ".format(self.num_classes)
        cmd += "--pretrained_model_dir {} ".format(local_dir)
        cmd += "--output_dir {}".format(output_dir)
        cmd = cmd.split(' ')
        logger.info("Distributed parameters processing cmd: {}".format(cmd))
        process = subprocess.Popen(cmd,
                                   stdout=sys.stdout,
                                   stderr=subprocess.STDOUT)
        process.wait()
        
        for file in os.listdir(local_dir):
            if "dist@" in file and "@rank@" in file:
                file = os.path.join(local_dir, file)
                os.remove(file)

        for file in os.listdir(output_dir):
            if "dist@" in file and "@rank@" in file:
                file = os.path.join(output_dir, file)
                shutil.move(file, local_dir)
        shutil.rmtree(output_dir)

    def _append_broadcast_ops(self, program):
        """
        Before test, we broadcast bathnorm-related parameters to all 
        other trainers from trainer-0.
        """
        bn_vars = [var for var in program.list_vars() 
                   if 'batch_norm' in var.name and var.persistable]
        block = program.current_block()
        for var in bn_vars:
            block._insert_op(
                0,
                type='c_broadcast',
                inputs={'X': var},
                outputs={'Out': var},
                attrs={'use_calc_stream': True})


    def load_checkpoint(self,
                        executor,
                        main_program,
                        use_per_trainer_checkpoint=False,
                        load_for_train=True):
        if use_per_trainer_checkpoint:
            checkpoint_dir = os.path.join(
                self.checkpoint_dir, str(self.trainer_id))
        else:
            checkpoint_dir = self.checkpoint_dir

        if self.fs_name is not None:
            ans = 'y'
            if os.path.exists(checkpoint_dir):
                ans = input("Downloading pretrained models, but the local "
                            "checkpoint directory ({}) exists, overwrite it "
                            "or not? [Y/N]".format(checkpoint_dir))

            if ans.lower() == 'y':
                if os.path.exists(checkpoint_dir):
                    logger.info("Using the local checkpoint directory.")
                    shutil.rmtree(checkpoint_dir)
                os.makedirs(checkpoint_dir)

                # sync all trainers to avoid loading checkpoints before 
                # parameters are downloaded
                file_name = os.path.join(checkpoint_dir, '.lock')
                if self.trainer_id == 0:
                    self.get_files_from_hdfs(checkpoint_dir)
                    with open(file_name, 'w') as f:
                        pass
                    time.sleep(10)
                    os.remove(file_name)     
                else:
                    while True:
                        if not os.path.exists(file_name):
                            time.sleep(1)
                        else:
                            break
        
        # Preporcess distributed parameters.
        file_name = os.path.join(checkpoint_dir, '.lock')
        distributed = self.loss_type in ["dist_softmax", "dist_arcface"]
        if load_for_train and self.trainer_id == 0 and distributed:
            self.preprocess_distributed_params(checkpoint_dir)
            with open(file_name, 'w') as f:
                pass
            time.sleep(10)
            os.remove(file_name)     
        elif load_for_train and distributed:
            # wait trainer_id (0) to complete
            while True:
                if not os.path.exists(file_name):
                    time.sleep(1)
                else:
                    break

        def if_exist(var):
            has_var = os.path.exists(os.path.join(checkpoint_dir, var.name))
            if has_var:
                print('var: %s found' % (var.name))
            return has_var

        fluid.io.load_vars(executor, checkpoint_dir, predicate=if_exist,
            main_program=main_program)

    def convert_for_prediction(self):
        model_name = self.model_name
        image_shape = [int(m) for m in self.image_shape]
        # model definition
        model = self.model
        if model is None:
            model = resnet.__dict__[model_name](emb_dim=self.emb_dim)
        main_program = self.train_program
        startup_program = self.startup_program
        with fluid.program_guard(main_program, startup_program):
            with fluid.unique_name.guard():
                image = fluid.layers.data(name='image',
                    shape=image_shape, dtype='float32')
                label = fluid.layers.data(name='label',
                    shape=[1], dtype='int64')

                emb = model.build_network(
                        input=image,
                        label=label,
                        is_train=False)

        gpu_id = int(os.getenv("FLAGS_selected_gpus", 0))
        place = fluid.CUDAPlace(gpu_id)
        exe = fluid.Executor(place)
        exe.run(startup_program)

        assert self.checkpoint_dir, "No checkpoint found for converting."
        self.load_checkpoint(executor=exe, main_program=main_program,
            load_for_train=False)

        assert self.model_save_dir, \
            "Does not set model_save_dir for inference model converting."
        if os.path.exists(self.model_save_dir):
            ans = input("model_save_dir for inference model ({}) exists, "
                        "overwrite it or not? [Y/N]".format(model_save_dir))
            if ans.lower() == 'n':
                logger.error("model_save_dir for inference model exists, "
                            "and cannot overwrite it.")
                exit()
            shutil.rmtree(self.model_save_dir)
        fluid.io.save_inference_model(self.model_save_dir,
                                      feeded_var_names=[image.name],
                                      target_vars=[emb],
                                      executor=exe,
                                      main_program=main_program)
        if self.fs_name:
            self.put_files_to_hdfs(model_save_dir)

    def predict(self):
        model_name = self.model_name
        image_shape = [int(m) for m in self.image_shape]
        # model definition
        model = self.model
        if model is None:
            model = resnet.__dict__[model_name](emb_dim=self.emb_dim)
        main_program = self.predict_program
        startup_program = self.startup_program
        with fluid.program_guard(main_program, startup_program):
            with fluid.unique_name.guard():
                image = fluid.layers.data(name='image',
                    shape=image_shape, dtype='float32')
                label = fluid.layers.data(name='label',
                    shape=[1], dtype='int64')

                emb = model.build_network(
                        input=image,
                        label=label,
                        is_train=False)

        gpu_id = int(os.getenv("FLAGS_selected_gpus", 0))
        place = fluid.CUDAPlace(gpu_id)
        exe = fluid.Executor(place)
        exe.run(startup_program)

        assert self.checkpoint_dir, "No checkpoint found for predicting."
        self.load_checkpoint(executor=exe, main_program=main_program,
            load_for_train=False)

        if self.train_reader is None:
            predict_reader = paddle.batch(reader.arc_train(
                self.dataset_dir, self.num_classes),
                batch_size=self.train_batch_size)
        else:
            predict_reader = self.train_reader

        feeder = fluid.DataFeeder(place=place,
            feed_list=['image', 'label'], program=main_program)
    
        fetch_list = [emb.name]
        for data in predict_reader():
            emb = exe.run(main_program, feed=feeder.feed(data),
                fetch_list=fetch_list, use_program_cache=True)
            print("emb: ", emb)

    def test(self, pass_id=0):
        self._check()

        trainer_id = self.trainer_id
        num_trainers = self.num_trainers
        worker_endpoints = os.getenv("PADDLE_TRAINER_ENDPOINTS")
        current_endpoint = os.getenv("PADDLE_CURRENT_ENDPOINT")

        emb, loss, acc1, acc5, _ = self.build_program(
            False, self.num_trainers > 1)

        config = dist_transpiler.DistributeTranspilerConfig()
        config.mode = "collective"
        config.collective_mode = "grad_allreduce"
        t = dist_transpiler.DistributeTranspiler(config=config)
        t.transpile(
                trainer_id=trainer_id,
                trainers=worker_endpoints,
                startup_program=self.startup_program,
                program=self.test_program,
                current_endpoint=current_endpoint)

        gpu_id = int(os.getenv("FLAGS_selected_gpus", 0))
        place = fluid.CUDAPlace(gpu_id)
        exe = fluid.Executor(place)
        exe.run(self.startup_program)

        test_list, test_name_list = reader.test(
            self.dataset_dir, self.val_targets)
        test_program = self.test_program
        #test_program = test_program._prune(emb)

        assert self.checkpoint_dir, "No checkpoint found for test."
        self.load_checkpoint(executor=exe, main_program=test_program,
            load_for_train=False)

        feeder = fluid.DataFeeder(place=place,
            feed_list=['image', 'label'], program=test_program)
        fetch_list = [emb.name]
        real_test_batch_size = self.global_test_batch_size

        test_start = time.time()
        for i in range(len(test_list)):
            data_list, issame_list = test_list[i]
            embeddings_list = []
            for j in xrange(len(data_list)):
                data = data_list[j]
                embeddings = None
                parallel_test_steps = data.shape[0] // real_test_batch_size
                beg = 0
                end = 0
                for idx in range(parallel_test_steps):
                    start = idx * real_test_batch_size
                    offset = trainer_id * self.test_batch_size
                    begin = start + offset
                    end = begin + self.test_batch_size
                    _data = []
                    for k in xrange(begin, end):
                        _data.append((data[k], 0))
                    assert len(_data) == self.test_batch_size
                    [_embeddings] = exe.run(test_program,
                        fetch_list = fetch_list, feed=feeder.feed(_data),
                        use_program_cache=True)
                    if embeddings is None:
                        embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
                    embeddings[start:start+real_test_batch_size, :] = _embeddings[:, :]
                beg = parallel_test_steps * real_test_batch_size
    
                while beg < data.shape[0]:
                    end = min(beg + self.test_batch_size, data.shape[0])
                    count = end - beg
                    _data = []
                    for k in xrange(end - self.test_batch_size, end):
                        _data.append((data[k], 0))
                    [_embeddings] = exe.run(test_program, 
                        fetch_list = fetch_list, feed=feeder.feed(_data),
                        use_program_cache=True)
                    _embeddings = _embeddings[0:self.test_batch_size,:]
                    embeddings[beg:end, :] = _embeddings[(self.test_batch_size-count):, :]
                    beg = end
                embeddings_list.append(embeddings)
    
            xnorm = 0.0
            xnorm_cnt = 0
            for embed in embeddings_list:
                xnorm += np.sqrt((embed * embed).sum(axis=1)).sum(axis=0)
                xnorm_cnt += embed.shape[0]
            xnorm /= xnorm_cnt
    
            embeddings = embeddings_list[0] + embeddings_list[1]
            embeddings = sklearn.preprocessing.normalize(embeddings)
            _, _, accuracy, val, val_std, far = evaluate(embeddings, issame_list, nrof_folds=10)
            acc, std = np.mean(accuracy), np.std(accuracy)
    
            print('[%s][%d]XNorm: %f' % (test_name_list[i], pass_id, xnorm))
            print('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (test_name_list[i], pass_id, acc, std))
            sys.stdout.flush()
        test_end = time.time()
        print("test time: {}".format(test_end - test_start))

    def train(self):
        self._check()

        trainer_id = self.trainer_id
        num_trainers = self.num_trainers
    
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        strategy = DistributedStrategy()
        strategy.mode = "collective"
        strategy.collective_mode = "grad_allreduce"
        self.fleet = fleet
        self.strategy = strategy
    
        train_emb, train_loss, train_acc1, train_acc5, optimizer = \
            self.build_program(True, False)
        if self.with_test:
            test_emb, test_loss, test_acc1, test_acc5, _ = \
                self.build_program(False, self.num_trainers > 1)
            test_list, test_name_list = reader.test(
                self.dataset_dir, self.val_targets)
            test_program = self.test_program
            self._append_broadcast_ops(test_program)
    
        global_lr = optimizer._global_learning_rate(
            program=self.train_program)
    
        origin_prog = fleet._origin_program
        train_prog = fleet.main_program
        if trainer_id == 0:
            with open('start.program', 'w') as fout:
                program_to_code(self.startup_program, fout, True)
            with open('main.program', 'w') as fout:
                program_to_code(train_prog, fout, True)
            with open('origin.program', 'w') as fout:
                program_to_code(origin_prog, fout, True)
            with open('test.program', 'w') as fout:
                program_to_code(test_program, fout, True)
    
        gpu_id = int(os.getenv("FLAGS_selected_gpus", 0))
        place = fluid.CUDAPlace(gpu_id)
        exe = fluid.Executor(place)
        exe.run(self.startup_program)
        
        if self.with_test:
            test_feeder = fluid.DataFeeder(place=place,
                feed_list=['image', 'label'], program=test_program)
            fetch_list_test = [test_emb.name] 
            real_test_batch_size = self.global_test_batch_size
    
        if self.checkpoint_dir:
            load_checkpoint = True
        else:
            load_checkpoint = False
        if load_checkpoint:
            self.load_checkpoint(executor=exe, main_program=origin_prog)
    
        if self.train_reader is None:
            train_reader = paddle.batch(reader.arc_train(
                self.dataset_dir, self.num_classes),
                batch_size=self.train_batch_size)
        else:
            train_reader = self.train_reader

        feeder = fluid.DataFeeder(place=place,
            feed_list=['image', 'label'], program=origin_prog)
    
        if self.calc_train_acc:
            fetch_list = [train_loss.name, global_lr.name,
                          train_acc1.name, train_acc5.name]
        else:
            fetch_list = [train_loss.name, global_lr.name]
    
        local_time = 0.0
        nsamples = 0
        inspect_steps = 200
        global_batch_size = self.global_train_batch_size
        for pass_id in range(self.train_epochs):
            train_info = [[], [], [], []]
            local_train_info = [[], [], [], []]
            for batch_id, data in enumerate(train_reader()):
                nsamples += global_batch_size
                t1 = time.time()
                if self.calc_train_acc:
                    loss, lr, acc1, acc5 = exe.run(train_prog,
                        feed=feeder.feed(data), fetch_list=fetch_list,
                        use_program_cache=True)
                else:
                    loss, lr, = exe.run(train_prog,
                        feed=feeder.feed(data), fetch_list=fetch_list,
                        use_program_cache=True)
                t2 = time.time()
                period = t2 - t1
                local_time += period
                train_info[0].append(np.array(loss)[0])
                train_info[1].append(np.array(lr)[0])
                local_train_info[0].append(np.array(loss)[0])
                local_train_info[1].append(np.array(lr)[0])
                if batch_id % inspect_steps == 0:
                    avg_loss = np.mean(local_train_info[0])
                    avg_lr = np.mean(local_train_info[1])
                    if self.calc_train_acc:
                        print("Pass:%d batch:%d lr:%f loss:%f qps:%.2f acc1:%.4f acc5:%.4f" % (
                            pass_id, batch_id, avg_lr, avg_loss, nsamples / local_time,
                            acc1, acc5))
                    else:
                        print("Pass:%d batch:%d lr:%f loss:%f qps:%.2f" % (
                            pass_id, batch_id, avg_lr, avg_loss, nsamples / local_time))
                    local_time = 0
                    nsamples = 0
                    local_train_info = [[], [], [], []]
    
            train_loss = np.array(train_info[0]).mean()
            print("End pass {0}, train_loss {1}".format(pass_id, train_loss))
            sys.stdout.flush()

            if self.with_test:
                test_start = time.time()
                for i in xrange(len(test_list)):
                    data_list, issame_list = test_list[i]
                    embeddings_list = []
                    for j in xrange(len(data_list)):
                        data = data_list[j]
                        embeddings = None
                        parallel_test_steps = data.shape[0] // real_test_batch_size
                        beg = 0
                        end = 0
                        for idx in range(parallel_test_steps):
                            start = idx * real_test_batch_size
                            offset = trainer_id * self.test_batch_size
                            begin = start + offset
                            end = begin + self.test_batch_size
                            _data = []
                            for k in xrange(begin, end):
                                _data.append((data[k], 0))
                            assert len(_data) == self.test_batch_size
                            [_embeddings] = exe.run(test_program,
                                fetch_list = fetch_list_test, feed=test_feeder.feed(_data),
                                use_program_cache=True)
                            if embeddings is None:
                                embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
                            embeddings[start:start+real_test_batch_size, :] = _embeddings[:, :]
                        beg = parallel_test_steps * real_test_batch_size
    
                        while beg < data.shape[0]:
                            end = min(beg + self.test_batch_size, data.shape[0])
                            count = end - beg
                            _data = []
                            for k in xrange(end - self.test_batch_size, end):
                                _data.append((data[k], 0))
                            [_embeddings] = exe.run(test_program, 
                                fetch_list = fetch_list_test, feed=test_feeder.feed(_data),
                                use_program_cache=True)
                            _embeddings = _embeddings[0:self.test_batch_size,:]
                            embeddings[beg:end, :] = _embeddings[(self.test_batch_size-count):, :]
                            beg = end
                        embeddings_list.append(embeddings)
    
                    xnorm = 0.0
                    xnorm_cnt = 0
                    for embed in embeddings_list:
                        xnorm += np.sqrt((embed * embed).sum(axis=1)).sum(axis=0)
                        xnorm_cnt += embed.shape[0]
                    xnorm /= xnorm_cnt
    
                    embeddings = embeddings_list[0] + embeddings_list[1]
                    embeddings = sklearn.preprocessing.normalize(embeddings)
                    _, _, accuracy, val, val_std, far = evaluate(embeddings, issame_list, nrof_folds=10)
                    acc, std = np.mean(accuracy), np.std(accuracy)
    
                    print('[%s][%d]XNorm: %f' % (test_name_list[i], pass_id, xnorm))
                    print('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (test_name_list[i], pass_id, acc, std))
                    sys.stdout.flush()
                test_end = time.time()
                print("test time: {}".format(test_end - test_start))
    
            #save model
            if self.model_save_dir:
                model_save_dir = os.path.join(
                    self.model_save_dir, str(pass_id))
                if not os.path.exists(model_save_dir):
                    # may be more than one processes trying 
                    # to create the directory
                    try:
                        os.makedirs(model_save_dir)
                    except OSError as exc:
                        if exc.errno != errno.EEXIST:
                            raise
                        pass
                if trainer_id == 0:
                    fluid.io.save_persistables(exe,
                        model_save_dir,
                        origin_prog)
                else:
                    def save_var(var):
                        to_save = "dist@" in var.name and '@rank@' in var.name
                        return to_save and var.persistable
                    fluid.io.save_vars(exe, model_save_dir,
                        origin_prog, predicate=save_var)

            #save training info
            if self.model_save_dir and trainer_id == 0:
                config_file = os.path.join(
                    self.model_save_dir, str(pass_id), 'meta.pickle')
                train_info = dict()
                train_info["pretrain_nranks"] = self.num_trainers
                train_info["emb_dim"] = self.emb_dim
                train_info['num_classes'] = self.num_classes
                with open(config_file, 'wb') as f:
                    pickle.dump(train_info, f)

        #upload model
        if self.model_save_dir and self.fs_name and trainer_id == 0:
            self.put_files_to_hdfs(self.model_save_dir)
            

if __name__ == '__main__':
    ins = Entry()
    ins.train()
