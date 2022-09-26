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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import collections
import paddle
from plsc.core import grad_sync, param_sync
from plsc.utils import io

from .utils import update_loss, update_metric, log_info

from plsc.utils import profiler
from plsc.utils import logger


def defualt_train_one_epoch(engine, epoch_id):
    tic = time.time()
    dev_id = paddle.distributed.ParallelEnv().dev_id
    for iter_id, batch in enumerate(engine.train_dataloader):
        for i in range(len(batch)):
            batch[i] = batch[i].cuda(dev_id)

        if iter_id >= engine.max_iter:
            break
        profiler.add_profiler_step(engine.config["profiler_options"])

        if iter_id == 5:
            for key in engine.time_info:
                engine.time_info[key].reset()
        engine.time_info["reader_cost"].update(time.time() - tic)

        batch_size = batch[0].shape[0]
        engine.global_step += 1

        # do forward and backward
        out, loss_dict = forward_backward(engine, batch)

        grad_sync(engine.optimizer.param_groups)

        # do unscale and step if using fp16 and not found nan/inf
        # otherwise do nothing
        engine.scaler.step(engine.optimizer)
        # do update loss scaling if using fp16
        # otherwise do nothing
        engine.scaler.update()
        # clear gradients
        engine.optimizer.clear_grad()

        if engine.lr_scheduler is not None and engine.lr_decay_unit == 'step':
            engine.lr_scheduler.step()

        # below code just for logging
        # update metric_for_logger
        update_metric(engine, out, batch, batch_size)
        # update_loss_for_logger
        update_loss(engine, loss_dict, batch_size)
        engine.time_info["batch_cost"].update(time.time() - tic)
        if iter_id % engine.print_batch_step == 0:
            log_info(engine, batch_size, epoch_id, iter_id)

        # eval model and save model if possible
        if engine.config["Global"]["eval_unit"] == 'step' and engine.config[
                "Global"]["eval_during_train"] and engine.global_step % engine.config[
                    "Global"][
                        "eval_interval"] == 0 and engine.eval_metric_func is not None:
            eval_metric_info = engine.eval(epoch_id)
            assert isinstance(eval_metric_info, dict)
            if eval_metric_info["metric"] > engine.best_metric["metric"]:
                if engine.config["Global"]["world_size"] > 1:
                    param_sync(engine.model, src_rank=eval_metric_info["rank"])
                engine.best_metric = eval_metric_info.copy()
                io.save_checkpoint(
                    engine.model,
                    engine.optimizer,
                    engine.lr_scheduler,
                    engine.best_metric,
                    engine.output_dir,
                    model_name=engine.config["Model"]["name"],
                    prefix="best_model",
                    max_num_checkpoint=engine.config["Global"][
                        "max_num_latest_checkpoint"], )
            logger.info("[Eval][Epoch {}][Step {}][best metric: {}]".format(
                epoch_id, engine.global_step, engine.best_metric["metric"]))
            logger.scaler(
                name="eval_metric",
                value=eval_metric_info["metric"],
                step=engine.global_step,
                writer=engine.vdl_writer)

            for key in engine.time_info:
                engine.time_info[key].reset()

        tic = time.time()


def forward_backward(engine, batch):
    # Gradient Merge(GuoxiaWang): Accumulate gradient over multiple 
    # steps to save on memory.

    assert batch[0].shape[
        0] % engine.accum_steps == 0, f'Bad accum_steps {engine.accum_steps} for batch size {batch[0].shape[0]}. This may be caused by two reasons: 1) the batch size setting is unreasonable and cannot be divisible, 2) drop_last in the sampler configuration is not set to True.'
    step_size = batch[0].shape[0] // engine.accum_steps

    final_loss_dict = collections.defaultdict(float)
    final_out = []

    for idx in range(engine.accum_steps):
        # do cast if using fp16 otherwise do nothing
        with paddle.amp.auto_cast(
                enable=engine.fp16,
                custom_white_list=engine.fp16_custom_white_list,
                custom_black_list=engine.fp16_custom_black_list,
                level=engine.fp16_level):

            inputs = {
                'data': batch[0][idx * step_size:(idx + 1) * step_size],
                'targets': batch[1][idx * step_size:(idx + 1) * step_size]
            }
            out = engine.model(inputs)
            targets = out['targets']
            out = out['logits']
            final_out.append(out)

        loss_dict = engine.train_loss_func(out, targets)

        for key in loss_dict:
            loss_dict[key] = loss_dict[key] / engine.accum_steps

            with paddle.no_grad():
                final_loss_dict[key] += loss_dict[key]

        # loss scaling if using fp16 otherwise do nothing
        scaled = engine.scaler.scale(loss_dict["loss"])
        #scaled = engine.scaler.scale(loss_dict["loss"]) * (1.0 / engine.config["Global"]["world_size"])
        scaled.backward()

    out = paddle.concat(final_out, axis=0)
    return out, final_loss_dict
