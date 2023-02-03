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
"""
Multi-task learning trainer
"""
import copy
import math
import numpy as np
import os
import random
import time

import paddle
import paddle.distributed as dist

from plsc.utils import logger, io
from plsc.utils.config import print_config
from plsc.models import build_model
from plsc.loss import build_mtl_loss
from plsc.core import GradScaler, param_sync
from plsc.core import grad_sync, recompute_warp
from plsc.optimizer import build_optimizer
from plsc.metric import build_metrics
from plsc.data import build_dataloader
from plsc.scheduler import build_lr_scheduler


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

        self.model_name = self.config["Model"].get("name", None)
        assert self.model_name, "model must be defined！"

        # init logger
        self.output_dir = self.config['Global']['output_dir']
        log_file = os.path.join(self.output_dir, self.model_name,
                                f"{self.mode}.log")
        logger.init_logger(log_file=log_file)

        # record
        print_config(self.config)

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

    def build_modules(self):
        # dataset
        if self.mode == "Train":
            for mode in ["Train", "Eval"]:
                data_loader = build_dataloader(
                    self.config["DataLoader"],
                    mode,
                    self.device,
                    worker_init_fn=self.worker_init_fn)
                setattr(self, f"{mode.lower()}_dataloader", data_loader)
            self.eval_metrics = build_metrics(self.config["Metric"]["Eval"])
        else:
            data_loader = build_dataloader(
                self.config["DataLoader"],
                self.mode,
                self.device,
                worker_init_fn=self.worker_init_fn)
            setattr(self, f"{self.mode.lower()}_dataloader", data_loader)

            metrics = build_metrics(self.config["Metric"][self.mode])
            setattr(self, f"{self.mode.lower()}_metrics", metrics)

        # build model
        self.model = build_model(
            self.config["Model"], task_names=self.task_names)
        if self.recompute:
            recompute_warp(self.model, **self.recompute_params)
        param_size, size_unit = self.params_counts(self.model)
        logger.info(
            f"The number of parameters is: {param_size:.3f}{size_unit}.")
        if self.dp:
            param_sync(self.model)
            logger.info("DDP model: sync parameters finished.")

        # build lr, opt, loss
        if self.mode == 'Train':
            # lr scheduler
            lr_config = copy.deepcopy(self.config.get("LRScheduler", None))
            self.lr_decay_unit = lr_config.get("decay_unit", "step")
            self.lr_scheduler = None
            if lr_config is not None:
                self.lr_scheduler = build_lr_scheduler(
                    lr_config, self.epochs, len(self.train_dataloader))
            # optimizer
            self.optimizer = build_optimizer(self.config["Optimizer"],
                                             self.lr_scheduler, self.model)

            self.loss_func = build_mtl_loss(self.task_names,
                                            self.config["Loss"][self.mode])

    def load_model(self):
        if self.checkpoint:
            io.load_checkpoint(self.checkpoint, self.model, self.optimizer,
                               self.scaler)
        elif self.pretrained_model:
            self.model.load_pretrained(self.pretrained_model, self.rank)

    def train(self):
        self.load_model()
        # train loop
        for epoch in range(self.epochs):
            self.train_one_epoch(epoch)
            # eval
            metric_results = {}
            if self.eval_during_train and self.eval_unit == "epoch" \
                    and (epoch + 1) % self.eval_interval == 0:
                metric_results = self.eval()
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
        for images, targets in self.train_dataloader:
            start = time.time()
            step += 1
            # compute loss
            with paddle.amp.auto_cast(self.fp16_params):
                logits = self.model(images)
                _, total_loss = self.loss_func(logits, targets)
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
        for images, targets in self.eval_dataloader:
            step += 1
            labels = targets["label"]
            tasks = targets["task"]
            logits = self.model(images)
            for idx, task_name in enumerate(self.task_names):
                cond = tasks == idx
                if not paddle.any(cond):
                    continue
                preds = logits[task_name][cond]
                labels = labels[cond]
                results[idx] = results.get(idx, {})
                task_metric = self.eval_metrics(preds, labels)
                for metric_name in task_metric:
                    results[idx][metric_name] = results[idx].get(metric_name,
                                                                 {})
                    for key in task_metric[metric_name]:
                        results[idx][metric_name][key] = \
                            results[idx][metric_name].get(key, 0) + task_metric[metric_name][key]
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
    def test(self):
        step = 0
        results = {}
        bs = {}
        self.model.eval()
        for images, targets in self.test_dataloader:
            step += 1
            labels = targets["label"]
            tasks = targets["task"]
            logits = self.model(images)
            for idx in range(len(tasks)):
                task_id = tasks[idx][0]
                preds_i = logits[self.task_names[task_id]]
                labels_i = labels[:, idx]
                results[idx] = results.get(idx, {})
                task_metric = self.test_metrics(preds_i, labels_i)
                for metric_name in task_metric:
                    results[idx][metric_name] = results[idx].get(metric_name,
                                                                 {})
                    for key in task_metric[metric_name]:
                        results[idx][metric_name][key] = \
                            results[idx][metric_name].get(key, 0) + task_metric[metric_name][key]
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
