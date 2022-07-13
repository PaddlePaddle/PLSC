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

import time
import numpy as np
import platform
import paddle

from plsc.utils.misc import AverageMeter
from plsc.utils import logger


def face_verification_eval(engine, epoch_id=0):
    output_info = dict()

    time_info = {
        "batch_cost": AverageMeter(
            "batch_cost", '.5f', postfix=" s,"),
        "reader_cost": AverageMeter(
            "reader_cost", ".5f", postfix=" s,"),
    }

    tic = time.time()

    total_samples = len(
        engine.eval_dataloader.
        dataset) if not engine.use_dali else engine.eval_dataloader.size
    max_iter = len(engine.eval_dataloader) - 1 if platform.system(
    ) == "Windows" else len(engine.eval_dataloader)

    embeddings_list = []
    issame_list = []
    dev_id = paddle.distributed.ParallelEnv().dev_id
    for iter_id, batch in enumerate(engine.eval_dataloader):
        for i in range(len(batch)):
            batch[i] = batch[i].cuda(dev_id)

        if iter_id >= max_iter:
            break
        if iter_id == 5:
            for key in time_info:
                time_info[key].reset()
        if engine.use_dali:
            batch = [
                paddle.to_tensor(batch[0]['data']),
                paddle.to_tensor(batch[0]['label'])
            ]
        time_info["reader_cost"].update(time.time() - tic)
        batch_size = batch[0].shape[0]

        # do cast if using fp16 otherwise do nothing
        with paddle.amp.auto_cast(
                enable=engine.fp16,
                custom_white_list=engine.fp16_custom_white_list,
                custom_black_list=engine.fp16_custom_black_list,
                level=engine.fp16_level):
            out = engine.model(batch[0])

        embedding = out.detach().cpu().numpy()
        embeddings_list.append(embedding)
        issame_list.append(batch[1].detach().cpu().numpy())

        time_info["batch_cost"].update(time.time() - tic)
        tic = time.time()

        if iter_id % engine.print_batch_step == 0:
            time_msg = "s, ".join([
                "{}: {:.5f}".format(key, time_info[key].avg)
                for key in time_info
            ])

            ips_msg = "ips: {:.5f} images/sec".format(
                batch_size / time_info["batch_cost"].avg)

            logger.info("[Eval][Epoch {}][Iter: {}/{}]{}, {}".format(
                epoch_id, iter_id,
                len(engine.eval_dataloader), time_msg, ips_msg))

    embeddings = np.vstack(embeddings_list)
    issames = np.concatenate(issame_list)

    # calc metric
    if engine.eval_metric_func is not None:
        output_info = engine.eval_metric_func(embeddings, issames)
        if engine.config["Global"]["world_size"] > 1:
            cur_metric = output_info["metric"]
            metric_tensor = paddle.to_tensor(
                output_info["metric"], dtype='float64')
            paddle.distributed.all_reduce(metric_tensor,
                                          paddle.distributed.ReduceOp.MAX)
            reduced_metric = metric_tensor.item()
            output_info["metric"] = metric_tensor.item()
            if reduced_metric == cur_metric:
                best_rank_id = paddle.to_tensor(
                    engine.config["Global"]["rank"], dtype='int32')
            else:
                best_rank_id = paddle.to_tensor(0, dtype='int32')
            paddle.distributed.all_reduce(best_rank_id,
                                          paddle.distributed.ReduceOp.MAX)
            output_info["rank"] = best_rank_id.item()

    if engine.use_dali:
        engine.eval_dataloader.reset()

    # do average
    for key in output_info:
        if isinstance(output_info[key], AverageMeter):
            output_info[key] = output_info[key].avg

    metric_msg = logger.dict_format(output_info)
    logger.info("[Eval][Epoch {}][Avg]{}".format(epoch_id, metric_msg))

    # do not try to save best eval.model
    if engine.eval_metric_func is None:
        return None

    output_info['epoch'] = epoch_id
    output_info['global_step'] = engine.global_step
    return output_info
