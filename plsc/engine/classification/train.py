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
import paddle
from plsc.core import grad_sync, param_sync

from .utils import update_loss, update_metric, log_info
from plsc.utils import profiler


def defualt_train_one_epoch(engine, epoch_id):
    tic = time.time()
    for iter_id, batch in enumerate(engine.train_dataloader):

        if iter_id >= engine.max_iter:
            break
        profiler.add_profiler_step(engine.config["profiler_options"])

        if iter_id == 5:
            for key in engine.time_info:
                engine.time_info[key].reset()
        engine.time_info["reader_cost"].update(time.time() - tic)
        if engine.use_dali:
            batch = [
                paddle.to_tensor(batch[0]['data']),
                paddle.to_tensor(batch[0]['label'])
            ]
        batch_size = batch[0].shape[0]
        engine.global_step += 1

        # do forward and backward
        out, loss_dict = forward_backward(engine, batch)

        grad_sync(engine.optimizer.param_groups)

        # default we use accum_steps=1
        if (iter_id + 1) % engine.accum_steps == 0:
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
        tic = time.time()


def forward_backward(engine, batch):
    # do cast if using fp16 otherwise do nothing
    with paddle.amp.auto_cast(
            enable=engine.fp16,
            custom_white_list=engine.fp16_custom_white_list,
            custom_black_list=engine.fp16_custom_black_list,
            level=engine.fp16_level):

        out = engine.model(batch[0])

    loss_dict = engine.train_loss_func(out, batch[1])

    # loss scaling if using fp16 otherwise do nothing
    scaled = engine.scaler.scale(loss_dict["loss"])
    scaled.backward()
    return out, loss_dict
