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

from __future__ import division
from __future__ import print_function

import errno
import json
import os
import math
import shutil
import subprocess
import sys
import tempfile
import time
import logging

import numpy as np
import paddle
import sklearn
import paddle.distributed.fleet as fleet
from paddle.optimizer import Optimizer

paddle.enable_static()

from . import config
from .models import DistributedClassificationOptimizer
from .models import base_model
from .models import resnet
from .utils import jpeg_reader as reader
from .utils.parameter_converter import rearrange_weight
from .utils.verification import evaluate
from .utils.input_field import InputField

log_handler = logging.StreamHandler()
log_format = logging.Formatter(
    '%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s')
log_handler.setFormatter(log_format)
logger = logging.getLogger(__name__)
logger.addHandler(log_handler)
logger.setLevel(logging.INFO)
logger.propagate = False


class Entry(object):
    """
    The class to encapsulate all operations.
    """

    def _check(self):
        """
        Check the validation of parameters.
        """
        supported_types = [
            "softmax",
            "arcface",
            "dist_softmax",
            "dist_arcface",
        ]
        assert self.loss_type in supported_types, \
            "All supported types are {}, but given {}.".format(
             supported_types, self.loss_type)

        if self.loss_type in ["dist_softmax", "dist_arcface"]:
            assert self.num_trainers > 1, \
                "At least 2 trainers are required for distributed fc-layer. " \
                "You can start your job using paddle.distributed.launch module."

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
        self.train_dataset = None
        self.test_dataset = None

        self.startup_program = paddle.static.Program()
        self.train_program = paddle.static.Program()
        self.test_program = paddle.static.Program()
        self.predict_program = paddle.static.Program()

        self.fs_name = None
        self.fs_ugi = None
        self.fs_dir_for_save = None
        self.fs_checkpoint_dir = None

        self.param_attr = None
        self.bias_attr = None

        self.has_run_train = False  # Whether has run training or not
        self.test_initialized = False
        self.cur_epoch = -1

        self.use_fp16 = False
        self.fp16_user_dict = None

        self.val_targets = self.config.val_targets
        self.dataset_dir = self.config.dataset_dir
        self.num_classes = self.config.num_classes
        self.loss_type = self.config.loss_type
        self.margin = self.config.margin
        self.scale = self.config.scale
        self.lr = self.config.lr
        self.lr_steps = self.config.lr_steps
        self.lr_scheduler = None
        self.train_image_num = self.config.train_image_num
        self.model_name = self.config.model_name
        self.emb_dim = self.config.emb_dim
        self.train_epochs = self.config.train_epochs
        self.checkpoint_dir = self.config.checkpoint_dir
        self.with_test = self.config.with_test
        self.model_save_dir = self.config.model_save_dir
        self.warmup_epochs = self.config.warmup_epochs
        self.calc_train_acc = False

        self.max_last_checkpoint_num = 5
        if self.checkpoint_dir:
            self.checkpoint_dir = os.path.abspath(self.checkpoint_dir)
        if self.model_save_dir:
            self.model_save_dir = os.path.abspath(self.model_save_dir)
        if self.dataset_dir:
            self.dataset_dir = os.path.abspath(self.dataset_dir)

        self.use_quant = False

        self.lr_decay_factor = 0.1
        self.log_period = 200
        self.test_period = 0
        self.cur_steps = 0

        self.input_info = [{
            'name': 'image',
            'shape': [-1, 3, 112, 112],
            'dtype': 'float32'
        }, {
            'name': 'label',
            'shape': [-1],
            'dtype': 'int64'
        }]
        self.input_field = None

        logger.info('=' * 30)
        logger.info("Default configuration:")
        for key in self.config:
            logger.info('\t' + str(key) + ": " + str(self.config[key]))
        logger.info('trainer_id: {}, num_trainers: {}'.format(trainer_id,
                                                              num_trainers))
        logger.info('default lr_decay_factor: {}'.format(self.lr_decay_factor))
        logger.info('default log period: {}'.format(self.log_period))
        logger.info('default test period: {}'.format(self.test_period))
        logger.info('=' * 30)

    def set_use_quant(self, quant):
        """
        Whether to use quantization
        """
        self.use_quant = quant

    def set_input_info(self, input_holder):
        """
        Set the information of inputs which is a list or tuple. Each element
        is a dict which contains the info of a input, including name, dtype
        and shape.
        """
        if not isinstance(input, (list, tuple)):
            raise ValueError("The type of 'input' must be a list or tuple.")

        has_label = False
        for element in input_holder:
            assert isinstance(element, dict), (
                "The type of elements for input must be dict")
            assert 'name' in element.keys(), (
                "Every element has to contain the key 'name'")
            assert 'shape' in element.keys(), (
                "Every element has to contain the key 'shape'")
            assert 'dtype' in element.keys(), (
                "Every element has to contain the key 'dtype'")
            if element['name'] == 'label':
                has_label = True
        assert has_label, "The input must contain a field named 'label'"

        self.input_holder = input_holder

    def set_val_targets(self, targets):
        """
        Set the names of validation datasets, separated by comma.
        """
        self.val_targets = targets
        logger.info("Set val_targets to {}.".format(targets))

    def set_train_batch_size(self, batch_size):
        self.train_batch_size = batch_size
        self.global_train_batch_size = batch_size * self.num_trainers
        logger.info("Set train batch size per trainer to {}.".format(
            batch_size))

    def set_log_period(self, period):
        self.log_period = period
        logger.info("Set log period to {}.".format(period))

    def set_test_period(self, period):
        self.test_period = period
        logger.info("Set test period to {}.".format(period))

    def set_lr_decay_factor(self, factor):
        self.lr_decay_factor = factor
        logger.info("Set lr decay factor to {}.".format(factor))

    def set_step_boundaries(self, boundaries):
        if not isinstance(boundaries, list):
            raise ValueError("The parameter must be of type list.")
        self.lr_steps = boundaries
        logger.info("Set step boundaries to {}.".format(boundaries))

    def set_mixed_precision(self,
                            use_fp16,
                            init_loss_scaling=1.0,
                            incr_every_n_steps=2000,
                            decr_every_n_nan_or_inf=2,
                            incr_ratio=2.0,
                            decr_ratio=0.5,
                            use_dynamic_loss_scaling=True,
                            amp_lists=None):
        """
        Whether to use mixed precision training.
        """
        self.use_fp16 = use_fp16
        self.fp16_user_dict = dict()
        self.fp16_user_dict['init_loss_scaling'] = init_loss_scaling
        self.fp16_user_dict['incr_every_n_steps'] = incr_every_n_steps
        self.fp16_user_dict[
            'decr_every_n_nan_or_inf'] = decr_every_n_nan_or_inf
        self.fp16_user_dict['incr_ratio'] = incr_ratio
        self.fp16_user_dict['decr_ratio'] = decr_ratio
        self.fp16_user_dict[
            'use_dynamic_loss_scaling'] = use_dynamic_loss_scaling
        self.fp16_user_dict['amp_lists'] = amp_lists
        logger.info("Use mixed precision training: {}.".format(use_fp16))
        for key in self.fp16_user_dict:
            logger.info("Set {} to {}.".format(key, self.fp16_user_dict[key]))

    def set_test_batch_size(self, batch_size):
        self.test_batch_size = batch_size
        self.global_test_batch_size = batch_size * self.num_trainers
        logger.info("Set test batch size per trainer to {}.".format(
            batch_size))

    def set_hdfs_info(self,
                      fs_name,
                      fs_ugi,
                      fs_dir_for_save=None,
                      fs_checkpoint_dir=None):
        """
        Set the info to download from or upload to hdfs filesystems.
        If the information is provided, we will download pretrained
        model from hdfs at the begining and upload pretrained models
        to hdfs at the end automatically.
        """
        self.fs_name = fs_name
        self.fs_ugi = fs_ugi
        self.fs_dir_for_save = fs_dir_for_save
        self.fs_checkpoint_dir = fs_checkpoint_dir
        logger.info("HDFS Info:")
        logger.info("\tfs_name: {}".format(fs_name))
        logger.info("\tfs_ugi: {}".format(fs_ugi))
        logger.info("\tfs dir for save: {}".format(self.fs_dir_for_save))
        logger.info("\tfs checkpoint dir: {}".format(self.fs_checkpoint_dir))

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
        logger.info("Calculating acc1 and acc5 during training: {}.".format(
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
        self.emb_dim = size
        logger.info("Set emb_dim to {}.".format(size))

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

    def set_max_last_checkpoint_num(self, num):
        """
        Set the max number of last checkpoint to keep.
        """
        self.max_last_checkpoint_num = num
        logger.info("Set max_last_checkpoint_num to {}.".format(num))

    def set_warmup_epochs(self, num):
        self.warmup_epochs = num
        logger.info("Set warmup_epochs to {}.".format(num))

    def set_loss_type(self, loss_type):
        supported_types = [
            "dist_softmax", "dist_arcface", "softmax", "arcface"
        ]
        if loss_type not in supported_types:
            raise ValueError("All supported loss types: {}".format(
                supported_types))
        self.loss_type = loss_type
        logger.info("Set loss_type to {}.".format(loss_type))

    def set_optimizer(self, optimizer):
        if not isinstance(optimizer, Optimizer):
            raise ValueError("Optimizer must be of type Optimizer")
        self.optimizer = optimizer
        logger.info("User manually set optimizer.")

    def set_with_test(self, with_test):
        self.with_test = with_test
        logger.info("Set with_test to {}.".format(with_test))

    def set_distfc_attr(self, param_attr=None, bias_attr=None):
        self.param_attr = param_attr
        logger.info("Set param_attr for distfc to {}.".format(self.param_attr))
        if self.bias_attr:
            self.bias_attr = bias_attr
            logger.info("Set bias_attr for distfc to {}.".format(
                self.bias_attr))

    def _set_info(self, key, value):
        if not hasattr(self, '_info'):
            self._info = {}
        self._info[key] = value

    def _get_info(self, key):
        if hasattr(self, '_info') and key in self._info:
            return self._info[key]
        return None

    def _get_optimizer(self):
        if not self.optimizer:
            bd = [step for step in self.lr_steps]
            start_lr = self.lr

            global_batch_size = self.global_train_batch_size
            train_image_num = self.train_image_num
            images_per_trainer = int(
                math.ceil(train_image_num * 1.0 / self.num_trainers))
            steps_per_pass = int(
                math.ceil(images_per_trainer * 1.0 / self.train_batch_size))
            logger.info("Steps per epoch: %d" % steps_per_pass)
            warmup_steps = steps_per_pass * self.warmup_epochs
            batch_denom = 1024
            base_lr = start_lr * global_batch_size / batch_denom
            lr_decay_factor = self.lr_decay_factor
            lr = [base_lr * (lr_decay_factor**i) for i in range(len(bd) + 1)]
            logger.info("LR boundaries: {}".format(bd))
            logger.info("lr_step: {}".format(lr))
            if self.warmup_epochs:
                self.lr_scheduler = paddle.optimizer.lr.LinearWarmup(
                    paddle.optimizer.lr.PiecewiseDecay(
                        boundaries=bd, values=lr),
                    warmup_steps,
                    start_lr,
                    base_lr)
            else:
                self.lr_scheduler = paddle.optimizer.lr.PiecewiseDecay(
                    boundaries=bd, values=lr)

            optimizer = paddle.optimizer.Momentum(
                learning_rate=self.lr_scheduler,
                momentum=0.9,
                weight_decay=paddle.regularizer.L2Decay(5e-4))
            self.optimizer = optimizer

        if self.loss_type in ["dist_softmax", "dist_arcface"]:
            self.optimizer = DistributedClassificationOptimizer(
                self.optimizer,
                self.train_batch_size,
                use_fp16=self.use_fp16,
                loss_type=self.loss_type,
                fp16_user_dict=self.fp16_user_dict)
        elif self.use_fp16:
            self.optimizer = paddle.static.amp.decorate(
                optimizer=self.optimizer,
                init_loss_scaling=self.fp16_user_dict['init_loss_scaling'],
                incr_every_n_steps=self.fp16_user_dict['incr_every_n_steps'],
                decr_every_n_nan_or_inf=self.fp16_user_dict[
                    'decr_every_n_nan_or_inf'],
                incr_ratio=self.fp16_user_dict['incr_ratio'],
                decr_ratio=self.fp16_user_dict['decr_ratio'],
                use_dynamic_loss_scaling=self.fp16_user_dict[
                    'use_dynamic_loss_scaling'],
                amp_lists=self.fp16_user_dict['amp_lists'])
        return self.optimizer

    def build_program(self, is_train=True, use_parallel_test=False):
        model_name = self.model_name
        assert not (is_train and use_parallel_test), \
            "is_train and use_parallel_test cannot be set simultaneously."

        trainer_id = self.trainer_id
        num_trainers = self.num_trainers

        # model definition
        model = self.model
        if model is None:
            model = resnet.__dict__[model_name](emb_dim=self.emb_dim)
        main_program = self.train_program if is_train else self.test_program
        startup_program = self.startup_program
        with paddle.static.program_guard(main_program, startup_program):
            with paddle.utils.unique_name.guard():
                input_field = InputField(self.input_info)
                input_field.build()
                self.input_field = input_field

                emb, loss, prob = model.get_output(
                    input=input_field,
                    num_classes=self.num_classes,
                    num_ranks=num_trainers,
                    rank_id=trainer_id,
                    is_train=is_train,
                    param_attr=self.param_attr,
                    bias_attr=self.bias_attr,
                    loss_type=self.loss_type,
                    margin=self.margin,
                    scale=self.scale)

                acc1 = None
                acc5 = None

                if self.loss_type in ["dist_softmax", "dist_arcface"]:
                    if self.calc_train_acc:
                        shard_prob = loss._get_info("shard_prob")

                        prob_list = []
                        paddle.distributed.all_gather(prob_list, shard_prob)
                        prob = paddle.concat(prob_list, axis=1)
                        label_list = []
                        paddle.distributed.all_gather(label_list,
                                                      input_field.label)
                        label_all = paddle.concat(label_list, axis=0)
                        acc1 = paddle.static.accuracy(
                            input=prob,
                            label=paddle.reshape(label_all, (-1, 1)),
                            k=1)
                        acc5 = paddle.static.accuracy(
                            input=prob,
                            label=paddle.reshape(label_all, (-1, 1)),
                            k=5)
                else:
                    if self.calc_train_acc:
                        acc1 = paddle.static.accuracy(
                            input=prob,
                            label=paddle.reshape(input_field.label, (-1, 1)),
                            k=1)
                        acc5 = paddle.static.accuracy(
                            input=prob,
                            label=paddle.reshape(input_field.label, (-1, 1)),
                            k=5)

                optimizer = None
                if is_train:
                    # initialize optimizer
                    optimizer = self._get_optimizer()
                    if self.num_trainers > 1:
                        dist_optimizer = fleet.distributed_optimizer(optimizer)
                        dist_optimizer.minimize(loss)
                    else:  # single card training
                        optimizer.minimize(loss)
                    if "dist" in self.loss_type or self.use_fp16:
                        optimizer = optimizer._optimizer
                elif use_parallel_test:
                    emb_list = []
                    paddle.distributed.all_gather(emb_list, emb)
                    emb = paddle.concat(emb_list, axis=0)
        return emb, loss, acc1, acc5, optimizer

    def get_files_from_hdfs(self):
        assert self.fs_checkpoint_dir, \
            logger.error("Please set the fs_checkpoint_dir paramerters for "
                         "set_hdfs_info to get models from hdfs.")
        self.fs_checkpoint_dir = os.path.join(self.fs_checkpoint_dir, '*')
        cmd = "hadoop fs -D fs.default.name="
        cmd += self.fs_name + " "
        cmd += "-D hadoop.job.ugi="
        cmd += self.fs_ugi + " "
        cmd += "-get " + self.fs_checkpoint_dir
        cmd += " " + self.checkpoint_dir
        logger.info("hdfs download cmd: {}".format(cmd))
        cmd = cmd.split(' ')
        process = subprocess.Popen(
            cmd, stdout=sys.stdout, stderr=subprocess.STDOUT)
        process.wait()

    def put_files_to_hdfs(self, local_dir):
        assert self.fs_dir_for_save, \
            logger.error("Please set fs_dir_for_save paramerter "
                         "for set_hdfs_info to save models to hdfs.")
        cmd = "hadoop fs -D fs.default.name="
        cmd += self.fs_name + " "
        cmd += "-D hadoop.job.ugi="
        cmd += self.fs_ugi + " "
        cmd += "-put " + local_dir
        cmd += " " + self.fs_dir_for_save
        logger.info("hdfs upload cmd: {}".format(cmd))
        cmd = cmd.split(' ')
        process = subprocess.Popen(
            cmd, stdout=sys.stdout, stderr=subprocess.STDOUT)
        process.wait()

    def _append_broadcast_ops(self, program):
        """
        Before test, we broadcast bathnorm-related parameters to all
        other trainers from trainer-0.
        """
        bn_vars = [
            var for var in program.list_vars()
            if 'batch_norm' in var.name and var.persistable
        ]
        block = program.current_block()
        for var in bn_vars:
            block._insert_op(
                0,
                type='c_broadcast',
                inputs={'X': var},
                outputs={'Out': var},
                attrs={'use_calc_stream': True})

    def save(self, program, epoch=0, for_train=True):
        if not self.model_save_dir:
            return

        trainer_id = self.trainer_id
        model_save_dir = os.path.join(self.model_save_dir, str(epoch))
        if not os.path.exists(model_save_dir):
            # may be more than one processes trying
            # to create the directory
            try:
                os.makedirs(model_save_dir)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
                pass

        param_state_dict = program.state_dict(mode='param')
        for name, param in param_state_dict.items():
            # for non dist param, we only save their at trainer 0,
            # but for dist param, we need to save their at all trainers
            if 'dist@' in name and '@rank@' in name or trainer_id == 0:
                paddle.save(param,
                            os.path.join(model_save_dir, name + '.pdparam'))

        if for_train:
            opt_state_dict = program.state_dict(mode='opt')
            for name, opt in opt_state_dict.items():
                # for non opt var, we only save their at trainer 0,
                # but for opt var, we need to save their at all trainers
                if 'dist@' in name and '@rank@' in name or trainer_id == 0:
                    paddle.save(opt,
                                os.path.join(model_save_dir, name + '.pdopt'))

            if trainer_id == 0:
                # save some extra info for resume
                # pretrain_nranks, emb_dim, num_classes are used for
                # re-split fc weight when gpu setting changed.
                # epoch use to restart.
                config_file = os.path.join(model_save_dir, 'meta.json')
                extra_info = dict()
                extra_info["pretrain_nranks"] = self.num_trainers
                extra_info["emb_dim"] = self.emb_dim
                extra_info['num_classes'] = self.num_classes
                extra_info['epoch'] = epoch
                extra_info['lr_state'] = self.lr_scheduler.state_dict()
                with open(config_file, 'w') as f:
                    json.dump(extra_info, f)

        logger.info("Save model to {}.".format(self.model_save_dir))
        if trainer_id == 0 and self.max_last_checkpoint_num > 0:
            for idx in range(-1, epoch - self.max_last_checkpoint_num + 1):
                path = os.path.join(self.model_save_dir, str(idx))
                if os.path.exists(path):
                    logger.info("Remove checkpoint {}.".format(path))
                    shutil.rmtree(path)

    def load(self, program, for_train=True):
        checkpoint_dir = os.path.abspath(self.checkpoint_dir)

        logger.info("Load checkpoint from '{}'. ".format(checkpoint_dir))
        if self.fs_name is not None:
            if os.path.exists(checkpoint_dir):
                logger.info("Local dir {} exists, we'll overwrite it.".format(
                    checkpoint_dir))

            # sync all trainers to avoid loading checkpoints before 
            # parameters are downloaded
            file_name = os.path.join(checkpoint_dir, '.lock')
            if self.trainer_id == 0:
                self.get_files_from_hdfs()
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

        state_dict = {}
        dist_weight_state_dict = {}
        dist_weight_velocity_state_dict = {}
        dist_bias_state_dict = {}
        dist_bias_velocity_state_dict = {}
        for path in os.listdir(checkpoint_dir):
            path = os.path.join(checkpoint_dir, path)
            if not os.path.isfile(path):
                continue

            basename = os.path.basename(path)
            name, ext = os.path.splitext(basename)

            if ext not in ['.pdopt', '.pdparam']:
                continue

            if not for_train and ext == '.pdopt':
                continue

            tensor = paddle.load(path, return_numpy=True)

            if 'dist@' in name and '@rank@' in name:
                if '.w' in name and 'velocity' not in name:
                    dist_weight_state_dict[name] = tensor
                elif '.w' in name and 'velocity' in name:
                    dist_weight_velocity_state_dict[name] = tensor
                elif '.b' in name and 'velocity' not in name:
                    dist_bias_state_dict[name] = tensor
                elif '.b' in name and 'velocity' in name:
                    dist_bias_velocity_state_dict[name] = tensor

            else:
                state_dict[name] = tensor

        distributed = self.loss_type in ["dist_softmax", "dist_arcface"]

        if for_train or distributed:
            meta_file = os.path.join(checkpoint_dir, 'meta.json')
            if not os.path.exists(meta_file):
                logger.error(
                    "Please make sure the checkpoint dir {} exists, and "
                    "parameters in that dir are validating.".format(
                        checkpoint_dir))
                exit()

            with open(meta_file, 'r') as handle:
                config = json.load(handle)

        # Preporcess distributed parameters.
        if distributed:
            pretrain_nranks = config['pretrain_nranks']
            assert pretrain_nranks > 0
            emb_dim = config['emb_dim']
            assert emb_dim == self.emb_dim
            num_classes = config['num_classes']
            assert num_classes == self.num_classes

            logger.info("Parameters for pre-training: pretrain_nranks ({}), "
                        "emb_dim ({}), and num_classes ({}).".format(
                            pretrain_nranks, emb_dim, num_classes))
            logger.info("Parameters for inference or fine-tuning: "
                        "nranks ({}).".format(self.num_trainers))

            trainer_id_str = '%05d' % self.trainer_id

            dist_weight_state_dict = rearrange_weight(
                dist_weight_state_dict, pretrain_nranks, self.num_trainers)
            dist_bias_state_dict = rearrange_weight(
                dist_bias_state_dict, pretrain_nranks, self.num_trainers)
            for name, value in dist_weight_state_dict.items():
                if trainer_id_str in name:
                    state_dict[name] = value
            for name, value in dist_bias_state_dict.items():
                if trainer_id_str in name:
                    state_dict[name] = value

            if for_train:
                dist_weight_velocity_state_dict = rearrange_weight(
                    dist_weight_velocity_state_dict, pretrain_nranks,
                    self.num_trainers)
                dist_bias_velocity_state_dict = rearrange_weight(
                    dist_bias_velocity_state_dict, pretrain_nranks,
                    self.num_trainers)
                for name, value in dist_weight_velocity_state_dict.items():
                    if trainer_id_str in name:
                        state_dict[name] = value
                for name, value in dist_bias_velocity_state_dict.items():
                    if trainer_id_str in name:
                        state_dict[name] = value
        if for_train:
            return {'state_dict': state_dict, 'extra_info': config}
        else:
            return {'state_dict': state_dict}

    def convert_for_prediction(self):
        model_name = self.model_name
        image_shape = [int(m) for m in self.image_shape]
        # model definition
        model = self.model
        if model is None:
            model = resnet.__dict__[model_name](emb_dim=self.emb_dim)
        main_program = self.predict_program
        startup_program = self.startup_program
        with paddle.static.program_guard(main_program, startup_program):
            with paddle.utils.unique_name.guard():
                input_field = InputField(self.input_info)
                input_field.build()

                emb = model.build_network(input=input_field, is_train=False)

        gpu_id = int(os.getenv("FLAGS_selected_gpus", 0))
        place = paddle.CUDAPlace(gpu_id)
        exe = paddle.static.Executor(place)
        exe.run(startup_program)

        assert self.checkpoint_dir, "No checkpoint found for converting."
        self.load(program=main_program, for_train=False)

        assert self.model_save_dir, \
            "Does not set model_save_dir for inference model converting."
        if os.path.exists(self.model_save_dir):
            logger.info("model_save_dir for inference model ({}) exists, "
                        "we will overwrite it.".format(self.model_save_dir))
            shutil.rmtree(self.model_save_dir)
        feed_var_names = []
        for name in input_field.feed_list_str:
            if name == "label": continue
            feed_var_names.append(name)
        paddle.static.save_inference_model(
            self.model_save_dir,
            feed_var_names, [emb],
            exe,
            program=main_program)
        if self.fs_name:
            self.put_files_to_hdfs(self.model_save_dir)

    def _run_test(self, exe, test_list, test_name_list, feeder, fetch_list):
        trainer_id = self.trainer_id
        real_test_batch_size = self.global_test_batch_size
        for i in range(len(test_list)):
            data_list, issame_list = test_list[i]
            embeddings_list = []
            # data_list[0] for normalize
            # data_list[1] for flip_left_right
            for j in range(len(data_list)):
                data = data_list[j]
                embeddings = None
                # For multi-card test, the dataset can be partitioned into two
                # part. For the first part, the total number of samples is
                # divisiable by the number of cards. And then, these samples
                # are split on different cards and tested parallely. For the
                # second part, these samples are tested on all cards but only
                # the result of the first card is used.

                # The number of steps for parallel test.
                parallel_test_steps = data.shape[0] // real_test_batch_size
                for idx in range(parallel_test_steps):
                    start = idx * real_test_batch_size
                    offset = trainer_id * self.test_batch_size
                    begin = start + offset
                    end = begin + self.test_batch_size
                    _data = []
                    for k in range(begin, end):
                        _data.append((data[k], 0))
                    assert len(_data) == self.test_batch_size
                    [_embeddings] = exe.run(self.test_program,
                                            fetch_list=fetch_list,
                                            feed=feeder.feed(_data),
                                            use_program_cache=True)
                    if embeddings is None:
                        embeddings = np.zeros((data.shape[0],
                                               _embeddings.shape[1]))
                    end = start + real_test_batch_size
                    embeddings[start:end, :] = _embeddings[:, :]
                beg = parallel_test_steps * real_test_batch_size

                while beg < data.shape[0]:
                    end = min(beg + self.test_batch_size, data.shape[0])
                    count = end - beg
                    _data = []
                    for k in range(end - self.test_batch_size, end):
                        _data.append((data[k], 0))
                    [_embeddings] = exe.run(self.test_program,
                                            fetch_list=fetch_list,
                                            feed=feeder.feed(_data),
                                            use_program_cache=True)
                    _embeddings = _embeddings[0:self.test_batch_size, :]
                    embeddings[beg:end, :] = _embeddings[(self.test_batch_size
                                                          - count):, :]
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
            _, _, accuracy, val, val_std, far = evaluate(
                embeddings, issame_list, nrof_folds=10)
            acc, std = np.mean(accuracy), np.std(accuracy)

            if self.cur_epoch >= 0:
                logger.info('[{}][{}][{}]XNorm: {:.5f}'.format(test_name_list[
                    i], self.cur_epoch, self.cur_steps, xnorm))
                logger.info('[{}][{}][{}]Accuracy-Flip: {:.5f}+-{:.5f}'.format(
                    test_name_list[
                        i], self.cur_epoch, self.cur_steps, acc, std))
            else:
                logger.info('[{}]XNorm: {:.5f}'.format(test_name_list[i],
                                                       xnorm))
                logger.info('[{}]Accuracy-Flip: {:.5f}+-{:.5f}'.format(
                    test_name_list[i], acc, std))
            sys.stdout.flush()

    def test(self):
        self._check()

        trainer_id = self.trainer_id
        num_trainers = self.num_trainers

        # if the test program is not built, which means that is the first time
        # to call the test method, we will first build the test program and
        # add ops to broadcast bn-related parameters from trainer 0 to other
        # trainers for distributed tests.
        if not self.test_initialized:
            emb, loss, _, _, _ = self.build_program(False,
                                                    self.num_trainers > 1)
            emb_name = emb.name
            assert self._get_info(emb_name) is None
            self._set_info('emb_name', emb.name)

            if num_trainers > 1 and self.has_run_train:
                self._append_broadcast_ops(self.test_program)

            if num_trainers > 1 and not self.has_run_train:
                worker_endpoints = os.getenv("PADDLE_TRAINER_ENDPOINTS")
                current_endpoint = os.getenv("PADDLE_CURRENT_ENDPOINT")

                #TODO how to transpile
                config = paddle.fluid.transpiler.DistributeTranspilerConfig()
                config.mode = "collective"
                config.collective_mode = "grad_allreduce"
                t = paddle.fluid.transpiler.DistributeTranspiler(config=config)
                t.transpile(
                    trainer_id=trainer_id,
                    trainers=worker_endpoints,
                    startup_program=self.startup_program,
                    program=self.test_program,
                    current_endpoint=current_endpoint)
        else:
            emb_name = self._get_info('emb_name')

        gpu_id = int(os.getenv("FLAGS_selected_gpus", 0))
        place = paddle.CUDAPlace(gpu_id)
        exe = paddle.static.Executor(place)
        if not self.has_run_train:
            exe.run(self.startup_program)

        if not self.test_dataset:
            test_reader = reader.test
        else:
            test_reader = self.test_reader
        if not self.test_initialized:
            test_list, test_name_list = test_reader(self.dataset_dir,
                                                    self.val_targets)
            assert self._get_info('test_list') is None
            assert self._get_info('test_name_list') is None
            self._set_info('test_list', test_list)
            self._set_info('test_name_list', test_name_list)
        else:
            test_list = self._get_info('test_list')
            test_name_list = self._get_info('test_name_list')

        test_program = self.test_program

        if not self.has_run_train:
            assert self.checkpoint_dir, "No checkpoint found for test."
            self.load_checkpoint(
                executor=exe, main_program=test_program, load_for_train=False)

        #TODO paddle.fluid.DataFeeder
        feeder = paddle.fluid.DataFeeder(
            place=place,
            feed_list=self.input_field.feed_list_str,
            program=test_program)
        fetch_list = [emb_name]

        self.test_initialized = True

        test_start = time.time()
        self._run_test(exe, test_list, test_name_list, feeder, fetch_list)
        test_end = time.time()
        logger.info("test time: {:.4f}".format(test_end - test_start))

    def train(self):
        self._check()
        self.has_run_train = True

        trainer_id = self.trainer_id
        num_trainers = self.num_trainers

        gpu_id = int(os.getenv("FLAGS_selected_gpus", 0))
        place = paddle.CUDAPlace(gpu_id)

        strategy = None
        if num_trainers > 1:
            strategy = fleet.DistributedStrategy()
            strategy.without_graph_optimization = True
            fleet.init(is_collective=True, strategy=strategy)

        emb, loss, acc1, acc5, optimizer = self.build_program(True, False)

        # define dataset
        if self.train_dataset is None:
            train_dataset = reader.TrainDataset(
                self.dataset_dir,
                self.num_classes,
                color_jitter=False,
                rotate=False,
                rand_mirror=True,
                normalize=True)
        else:
            train_dataset = self.train_dataset

        dataloader = paddle.io.DataLoader(
            train_dataset,
            feed_list=self.input_field.feed_list,
            places=place,
            return_list=False,
            batch_size=self.train_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4)

        global_lr = optimizer._global_learning_rate(program=self.train_program)

        origin_prog = self.train_program
        train_prog = self.train_program

        exe = paddle.static.Executor(place)
        exe.run(self.startup_program)

        if self.checkpoint_dir:
            load_checkpoint = True
        else:
            load_checkpoint = False

        start_epoch = 0
        self.cur_steps = 0
        if load_checkpoint:
            checkpoint = self.load(program=origin_prog, for_train=True)
            origin_prog.set_state_dict(checkpoint['state_dict'])
            start_epoch = checkpoint['extra_info']['epoch'] + 1
            lr_state = checkpoint['extra_info']['lr_state']
            # there last_epoch means last_step in step style
            self.cur_steps = lr_state['last_epoch']
            self.lr_scheduler.set_state_dict(lr_state)

        if self.calc_train_acc:
            fetch_list = [loss.name, global_lr.name, acc1.name, acc5.name]
        else:
            fetch_list = [loss.name, global_lr.name]

        local_time = 0.0
        nsamples = 0
        inspect_steps = self.log_period
        global_batch_size = self.global_train_batch_size
        for epoch in range(start_epoch, self.train_epochs):
            self.cur_epoch = epoch
            train_info = [[], [], [], []]
            local_train_info = [[], [], [], []]
            for batch_id, data in enumerate(dataloader):
                self.cur_steps += 1
                nsamples += global_batch_size
                t1 = time.time()
                acc1 = None
                acc5 = None
                if self.calc_train_acc:
                    loss, lr, acc1, acc5 = exe.run(train_prog,
                                                   feed=data,
                                                   fetch_list=fetch_list,
                                                   use_program_cache=True)
                else:
                    loss, lr = exe.run(train_prog,
                                       feed=data,
                                       fetch_list=fetch_list,
                                       use_program_cache=True)
                self.lr_scheduler.step()
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
                    speed = nsamples / local_time
                    if self.calc_train_acc:
                        logger.info("Pass:{} batch:{} lr:{:.8f} loss:{:.6f} "
                                    "qps:{:.2f} acc1:{:.6f} acc5:{:.6f}".
                                    format(epoch, batch_id, avg_lr, avg_loss,
                                           speed, acc1[0], acc5[0]))
                    else:
                        logger.info(
                            "Pass:{} batch:{} lr:{:.8f} loss:{:.6f} "
                            "qps:{:.2f}".format(epoch, batch_id, avg_lr,
                                                avg_loss, speed))
                    local_time = 0
                    nsamples = 0
                    local_train_info = [[], [], [], []]

                if self.test_period > 0 and self.cur_steps % self.test_period == 0:
                    if self.with_test:
                        self.test()

            train_loss = np.array(train_info[0]).mean()
            logger.info("End pass {}, train_loss {:.6f}".format(epoch,
                                                                train_loss))
            sys.stdout.flush()

            # save model
            self.save(origin_prog, epoch=epoch)

        # upload model
        if self.model_save_dir and self.fs_name and trainer_id == 0:
            self.put_files_to_hdfs(self.model_save_dir)


if __name__ == '__main__':
    ins = Entry()
    ins.train()
