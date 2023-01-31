# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import math
import numpy as np
import os
import random
import time
from typing import Dict
from collections import defaultdict

import paddle
import paddle.distributed as dist
from paddle.io import DataLoader, DistributedBatchSampler
# from paddle.distributed.fleet.utils.hybrid_parallel_util
# import fused_allreduce_gradients

from plsc.engine.engine import Engine
from plsc.utils import logger
from plsc.utils import io
from plsc.models.multi_task.MTLModel import MTLModel
from plsc.loss.MTLoss import MTLoss
from plsc.core.param_fuse import get_fused_params
from plsc.core import recompute_warp, GradScaler, param_sync, grad_sync
from plsc.models.multi_task.ResNet_backbone import IResNet18, IResNet50
from plsc.models.multi_task.head import TaskBlock
from plsc.scheduler import ViTLRScheduler
from plsc.optimizer import AdamW, ClipGradByGlobalNorm
from plsc.data.sampler.mtl_sampler import MTLSampler
from plsc.metric.metrics import TopkAcc
from plsc.data.dataset.mtl_dataset import SingleTaskDataset, \
    MultiTaskDataset, ConcatDataset


class MTLEngine(object):
    def __init__(self, config, mode="Train"):
        self.mode = mode
        self.finetune = False
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.config = config
        self.parse_config()
        self.build_modules()

    @staticmethod
    def params_counts(model):
        n_parameters = sum(p.numel() for p in model.parameters()
                           if not p.stop_gradient).item()
        i = int(math.log(n_parameters, 10) // 3)
        size_unit = ['', 'K', 'M', 'B', 'T', 'Q']
        param_size = n_parameters / math.pow(1000, i)
        return param_size, size_unit[i]

    def _init_worker(self, worker_id):
        """ set seed in subproces for dataloader when num_workers > 0"""
        if self.seed:
            np.random.seed(self.seed + worker_id)
            random.seed(self.seed + worker_id)

    def parse_config(self):

        # parse global params
        for key in self.config["Global"]:
            setattr(self, key, self.config["Global"][key])

        # distillation and get model name
        if self.config.get("Distillation", None):
            student_model = self.config["Model"].get("Student", None)
            teacher_model = self.config["Model"].get("Teacher", None)
            assert student_model and teacher_model, "Teacher and Student model must be defined！"
            self.model_name = student_model.get("name", None)

            self.distillation = self.config["Distillation"]["Enabled"]
            self.soft_weight = self.config["Distillation"]["soft_weight"]
        else:
            self.distillation = False
            if self.config["Model"].get("Teacher", None):
                self.model_name = self.config["Model"]["Teacher"].get("name",
                                                                      None)
            else:
                self.model_name = self.config["Model"].get("name", None)
        assert self.model_name, "model must be defined！"

        # init logger
        self.output_dir = self.config['Global']['output_dir']
        log_file = os.path.join(self.output_dir, self.model_name,
                                f"{self.mode}.log")
        logger.init_logger(log_file=log_file)

        # set device
        assert self.config["Global"]["device"] in ["cpu", "gpu", "xpu", "npu"]
        self.device = paddle.set_device(self.config["Global"]["device"])
        logger.info('train with paddle {}, commit id {} and device {}'.format(
            paddle.__version__, paddle.__git_commit__[:8], self.device))

        # set seed
        self.seed = self.config["Global"].get("seed", False)
        if self.seed:
            assert isinstance(self.seed, int), "The 'seed' must be a integer!"
            self.seed += self.rank
            paddle.seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)
        self.worker_init_fn = self._init_worker if self.seed else None

        # distributed strategy
        cfg_dist = self.config.get("DistributedStrategy", None)
        if cfg_dist.get("data_parallel", None):
            self.dp = True
            dist.init_parallel_env()

        self.recompute = False
        self.recompute_params = {}
        if cfg_dist.get("recompute", None):
            self.recompute = True
            self.recompute_params = cfg_dist["recompute"]

        # amp
        cfg_fp16 = self.config.get("FP16", False)
        self.fp16_params = {"enable": False}
        if cfg_fp16:
            self.fp16_params["level"] = cfg_fp16.get("level", "O1")
            if self.fp16_params["level"] != 'O0':
                self.fp16_params["enable"] = True
            cfg_scaler = cfg_fp16.get("GradScaler", {})
            self.scaler = GradScaler(self.fp16_params["enable"], **cfg_scaler)
            self.fp16_params["custom_white_list"] = cfg_fp16.get(
                "fp16_custom_white_list", None)
            self.fp16_params["custom_black_list"] = cfg_fp16.get(
                "fp16_custom_black_list", None)
        # record
        self.print_config()

    def print_config(self):
        def print_params_dic(params_dic):
            for key in params_dic:
                logger.info(f"{key}: {params_dic[key]}")

        logger.info("=" * 16 + " params " + "=" * 16)
        print_params_dic(self.config)
        logger.info("=" * 40)

    def build_modules(self):
        # dataset
        for mode in ["Train", "Eval", "Test"]:
            self.build_mtl_dataset(mode)
            self.build_mtl_sampler(mode)
            self.build_mtl_loader(mode)
            self.build_metrics(mode)

        # build model
        if self.distillation:
            self.build_model(opt="Teacher")
            self.build_model(opt="Student")
            self.model = self.student_model
        else:
            self.build_model(opt="Teacher")
            self.model = self.teacher_model
        if self.dp:
            param_sync(self.model)
            # model = paddle.DataParallel(model, find_unused_parameters=True)
            logger.info("DDP model: sync parameters finished")

        # build lr, opt, loss
        if self.mode == 'Train':
            self.build_lr_scheduler()
            self.build_optimizer()
            self.build_loss()
            if self.distillation:
                self.build_distill_loss()

    def build_mtl_dataset(self, mode):
        # multi-task dataset
        cfg_ds_list = self.config["DataLoader"][mode][
            "dataset"]  # dataset list
        datasets = []
        all_sample_ratio = []
        for cfg_ds_item in cfg_ds_list:
            dataset_name = list(cfg_ds_item.keys())[0]
            cfg_ds = cfg_ds_item[dataset_name]
            label_path_list = cfg_ds["cls_label_path"]
            if not isinstance(cfg_ds["cls_label_path"], list):
                label_path_list = [cfg_ds["cls_label_path"]]
            assert len(label_path_list) == len(cfg_ds["task_ids"]), \
                "lenght of label_path_list must be equal to task_names"

            sample_ratio = cfg_ds.get("sample_ratio", 1.)
            if not isinstance(sample_ratio, list):
                sample_ratio = [sample_ratio] * len(label_path_list)
            all_sample_ratio += sample_ratio

            for i in range(len(label_path_list)):
                st_dataset = eval(cfg_ds["name"])(
                    cfg_ds["task_ids"][i], cfg_ds["data_root"],
                    label_path_list[i], cfg_ds["transform_ops"])
                datasets.append(st_dataset)
        if len(datasets) >= 1:
            dataset = ConcatDataset(datasets, dataset_ratio=all_sample_ratio)
        else:
            dataset = datasets[0]
        setattr(self, f"{mode.lower()}_dataset", dataset)
        logger.debug(f"Build {mode} dataset succeed.")

    def build_mtl_sampler(self, mode):
        # multi-task sampler
        cfg_sampler = self.config["DataLoader"][mode]["sampler"]
        sampler_name = cfg_sampler.pop("name")
        dataset = getattr(self, f"{mode.lower()}_dataset")
        batch_sampler = eval(sampler_name)(dataset, **cfg_sampler)
        logger.debug("build batch_sampler({}) success...".format(sampler_name))
        setattr(self, f"{mode.lower()}_sampler", batch_sampler)
        logger.debug(f"Build {mode} sampler succeed.")

    def build_mtl_loader(self, mode):
        # multi-task data loader
        config_loader = self.config["DataLoader"][mode]["loader"]
        dataset = getattr(self, f"{mode.lower()}_dataset")
        sampler = getattr(self, f"{mode.lower()}_sampler")
        data_loader = DataLoader(
            dataset=dataset,
            places=self.device,
            num_workers=config_loader.num_workers,
            return_list=True,
            use_shared_memory=config_loader.use_shared_memory,
            batch_sampler=sampler,
            worker_init_fn=self.worker_init_fn)
        setattr(self, f"{mode.lower()}_dataloader", data_loader)
        logger.debug(f"Build {mode} dataloader succeed.")

    def build_model(self, opt=None):
        model_config = copy.deepcopy(self.config["Model"])
        if model_config.get(opt, None):
            model_config = model_config[opt]
        # structure
        model_name = model_config["name"]
        # backbone
        config_backbone = model_config["backbone"]
        backbone_name = config_backbone.pop("name")
        backbone = eval(backbone_name)(**config_backbone)
        # head
        config_heads = model_config["heads"]
        head_dic = {}
        for head_item in config_heads:
            cfg_head = copy.deepcopy(head_item[list(head_item.keys())[0]])
            task_ids = cfg_head.pop("task_ids")
            class_nums = cfg_head.pop("class_nums")
            head_class = cfg_head.pop("name")
            if not isinstance(head_class, list):
                head_class = [head_class] * len(task_ids)
            if not isinstance(class_nums, list):
                class_nums = [class_nums] * len(task_ids)
            for i, task_id in enumerate(task_ids):
                head_dic[self.task_names[task_id]] = (eval(head_class[i])(
                    class_num=class_nums[i], **cfg_head))
        # merge
        model = eval(model_name)(backbone, head_dic, self.recompute,
                                 self.recompute_params)
        setattr(self, f"{opt.lower()}_model", model)
        param_size, size_unit = self.params_counts(model)
        logger.info(
            f"Build {opt} model succeed, the number of parameters is: {param_size:.3f}{size_unit}."
        )

    def build_loss(self):
        cfg_loss = self.config["Loss"][self.mode]
        self.loss_func = MTLoss(self.task_names, cfg_loss)
        logger.debug(f"build {self.mode} loss {self.loss_func} success.")

    def build_lr_scheduler(self):
        lr_config = copy.deepcopy(self.config.get("LRScheduler", None))
        self.lr_decay_unit = lr_config.get("decay_unit", "step")
        lr_config.update({
            "epochs": self.epochs,
            "step_each_epoch": len(self.train_dataloader)
        })
        if "name" in lr_config:
            lr_name = lr_config.pop("name")
            lr = eval(lr_name)(**lr_config)
            if isinstance(lr, paddle.optimizer.lr.LRScheduler):
                self.lr_scheduler = lr
            else:
                self.lr_scheduler = lr()
        else:
            self.lr_scheduler = lr_config["learning_rate"]
        logger.debug("build lr ({}) success..".format(self.lr_scheduler))

    def build_optimizer(self):
        opt_config = copy.deepcopy(self.config["Optimizer"])
        grad_clip = None
        grad_clip_config = opt_config.pop('grad_clip', None)
        if grad_clip_config is not None:
            grad_clip_name = grad_clip_config.pop('name',
                                                  'ClipGradByGlobalNorm')
            grad_clip = eval(grad_clip_name)(**grad_clip_config)
        no_weight_decay_name = opt_config.pop('no_weight_decay_name', [])

        param_group = defaultdict(list)
        for n, p in self.model.named_parameters():
            state = copy.deepcopy(p.__dict__)
            if any(nd in n for nd in no_weight_decay_name):
                state['no_weight_decay'] = True
            param_group[str(state)].append(p)

        # fuse params
        for key in param_group:
            if 'gpu' not in paddle.get_device():
                continue
            if "'is_distributed': True" in key:
                continue
            if "'has_sparse_grad': True" in key:
                continue
            param_group[key] = get_fused_params(param_group[key])

        # bulid optimizer params
        params = []
        for key in param_group:
            group = {'params': param_group[key]}

            if "'is_distributed': True" in key:
                group['is_distributed'] = True

            if 'no_weight_decay' in key:
                group['weight_decay'] = 0.0

            params.append(group)

        optim_name = opt_config.pop('name')
        self.optimizer = eval(optim_name)(params,
                                          lr=self.lr_scheduler,
                                          grad_clip=grad_clip,
                                          **opt_config)

        logger.debug("build optimizer ({}) success..".format(self.optimizer))

    def build_metrics(self, mode):
        cfg_metric = self.config.get("Metric", None)
        metrics = []
        if cfg_metric is not None:
            metric_func_list = cfg_metric[mode]
            for item in metric_func_list:
                func_name = list(item.keys())[0]
                metrics.append(eval(func_name)(**item[func_name]))
        setattr(self, f"{mode.lower()}_metrics", metrics)
        logger.debug(f"Build {mode} metrics succeed.")

    def build_distill_loss(self):
        cfg_loss = self.config["Distillation"].get("soft_loss", None)
        assert cfg_loss is not None, "distillation loss should not be None"
        self.distill_loss = MTLoss(self.task_names, cfg_loss)
        logger.debug("build distill loss success.")

    def load_model(self):
        if self.checkpoint:
            io.load_checkpoint(self.checkpoint, self.model, self.optimizer,
                               self.scaler)
        elif self.pretrained_model:
            self.model.load_pretrained(self.pretrained_model, self.rank)
        if self.distillation:
            teacher_model_path = self.config["Global"].get(
                "teacher_checkpoint", None)
            assert teacher_model_path is not None, "Lack of teacher checkpoint, " \
                                                   "which must be loaded first in distillation mode "
            self.teacher_model.load_pretrained(teacher_model_path, self.rank)
            self.teacher_model.eval()
            logger.info(f"Teacher model initialized.")

    def train(self):
        self.load_model()
        # train loop
        for epoch in range(self.epochs):
            self.train_one_epoch(epoch)
            # eval
            metric_results = {}
            if self.eval_during_train and self.eval_unit == "epoch" \
                    and (epoch + 1) % self.eval_interval == 0:
                metric_results = self.test()
            # save model
            if (epoch + 1) % self.save_interval == 0 or (epoch + 1
                                                         ) == self.epochs:
                model_prefix = "final" if (
                    epoch + 1) == self.epochs else f"model_epoch{epoch}"
                self.save_model(model_prefix, metric_results)
            # update lr
            if self.lr_decay_unit == "epoch":
                self.optimizer.lr_step()

    def train_one_epoch(self, epoch):
        step = 0
        avg_loss = 0  # average loss in the lasted `self.print_batch_step` steps
        for images, labels, tasks in self.train_dataloader:
            start = time.time()
            step += 1
            # compute loss
            with paddle.amp.auto_cast(self.fp16_params):
                logits = self.model(images)
                _, total_loss = self.loss_func(logits, labels, tasks)
                if self.distillation:
                    with paddle.no_grad():
                        teacher_logits = self.teacher_model(images)
                    _, total_soft_loss = self.distill_loss(
                        logits, teacher_logits, tasks)
                    total_loss = self.soft_weight * total_soft_loss + (
                        1 - self.soft_weight) * total_loss
            scaled = self.scaler.scale(total_loss)
            scaled.backward()
            grad_sync(self.optimizer.param_groups)
            # update params
            if (step + 1) % self.accum_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.clear_grad()
            if self.lr_decay_unit == "step":
                self.optimizer.lr_step()
            # show loss
            avg_loss += total_loss.cpu().numpy()[0]
            if (step + 1) % self.print_batch_step == 0:
                logger.info(f"epoch: {epoch}, step: {step}, "
                            f"total loss: {avg_loss / self.print_batch_step}")
                avg_loss = 0
            end = time.time()
            logger.debug(f"one step time = {(end - start): .3f}s")

    def save_model(self, model_prefix, metric_results=None):
        io.save_checkpoint(
            self.model,
            self.optimizer,
            self.scaler,
            metric_results,
            self.output_dir,
            model_name=self.model_name,
            prefix=model_prefix,
            max_num_checkpoint=self.max_num_latest_checkpoint)

    @paddle.no_grad()
    def eval(self):
        step = 0
        results = {}
        bs = {}
        self.model.eval()
        for images, labels, tasks in self.eval_dataloader:
            step += 1
            logits = self.model(images)
            for idx, task_name in enumerate(self.task_names):
                cond = tasks == idx
                if not paddle.any(cond):
                    continue
                preds = logits[task_name][cond]
                labels = labels[cond]
                for eval_metric in self.eval_metrics:
                    task_metric = eval_metric(preds, labels)
                    metric_name = str(eval_metric).replace("()", "")
                    results[idx] = results.get(idx, {metric_name: {}})
                    for key in task_metric:
                        results[idx][metric_name][key] = \
                            results[idx][metric_name].get(key, 0) + task_metric[key]
                bs[idx] = bs.get(idx, 0) + 1
        self.model.train()
        for idx in results:
            for metric in results[idx]:
                for key in results[idx][metric]:
                    results[idx][metric][key] /= bs[idx]
        return results

    @paddle.no_grad()
    def test(self):
        step = 0
        results = {}
        bs = {}
        self.model.eval()
        for images, targets, tasks in self.test_dataloader:
            step += 1
            logits = self.model(images)
            for idx, task_id in enumerate(tasks[0]):
                preds = logits[self.task_names[task_id]]
                labels = targets[:, idx]
                for metric in self.test_metrics:
                    task_metric = metric(preds, labels)
                    metric_name = str(metric).replace("()", "")
                    results[idx] = results.get(idx, {metric_name: {}})
                    for key in task_metric:
                        results[idx][metric_name][key] = \
                            results[idx][metric_name].get(key, 0) + task_metric[key]
                bs[idx] = bs.get(idx, 0) + 1
        self.model.train()
        for idx in results:
            for metric in results[idx]:
                for key in results[idx][metric]:
                    results[idx][metric][key] /= bs[idx]
        for task_id in results:
            logger.info(f"metrics - task{task_id}: {results[task_id]}")
        return results

    @paddle.no_grad()
    def export(self):
        assert self.mode in ["Export", "export"]
        assert self.config.get("Export", None) is not None
        assert self.pretrained_model is not None
        self.model.eval()
        path = os.path.join(self.output_dir, self.model_name)
        io.export(self.config["Export"], self.model, path)
