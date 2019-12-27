#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import warnings
import os
import six
import logging
import argparse
import shutil
import json

import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.fluid.transpiler.details import program_to_code


logging.basicConfig(level=logging.INFO,
    format='[%(levelname)s %(asctime)s line:%(lineno)d] %(message)s',
    datefmt='%d %b %Y %H:%M:%S')
logger = logging.getLogger()


parser = argparse.ArgumentParser(description="""
    Tool to convert pretrained distributed fc parameters for inference.

    Note that the number of ranks or GPUs for inference can be different 
    from that for pretraining.""")

parser.add_argument("--name_feature",
                    type=str,
                    default="@rank@",
                    help="Feature for names of distributed fc parameters. "
                         "For example, by default the name for the "
                         "distributed fc weight parameter is like "
                         "dist@xxx@rank@id.w_0 where xxx is softmax or arcface "
                         "depending on the loss types used and rank_id is the "
                         "rank_id generating this parameter, and hence the "
                         "feature cloud be @rank@.")
parser.add_argument("--pretrain_nranks",
                    type=int,
                    default=-1,
                    help="Number of ranks (GPUs) for pre-training.")
parser.add_argument("--nranks",
                    type=int,
                    required=True,
                    help="Number of ranks (GPUs) for inference or finetuning.")
parser.add_argument("--num_classes",
                    type=int,
                    default=-1,
                    help="Number of classes for classification.")
parser.add_argument("--emb_dim",
                    type=int,
                    default=-1,
                    help="Embedding dim.")
parser.add_argument("--pretrained_model_dir",
                    type=str,
                    required=True,
                    default=None,
                    help="Directory for pretrained model.")
parser.add_argument("--output_dir",
                    type=str,
                    required=True,
                    default=None,
                    help="Directory for output.")
args = parser.parse_args()


def load_config(args):
    """
    Load config file which contains the following information for pretraining:
        1. pretrain_nranks (int): number of ranks for pretraining;
        2. emb_dim (int): embedding dim for pretraining;
        3. num_classes (int): number of classes for classification.
    """
    meta_file = os.path.join(args.pretrained_model_dir, 'meta.json')
    if not os.path.exists(meta_file):
        if args.pretrain_nranks < 0 or args.emb_dim < 0 or args.num_classes < 0:
            logger.error("Meta file does not exist, you have to set "
                         "'--pretrain_nranks', '--emb_dim' and '--num_classes "
                         "parameters manually.")
            exit()
        logger.debug("Meta file does not exist, make sure you have correctly "
                     "set --pretrain_nranks ({}), --emb_dim ({}) and "
                     "--num_classes ({}) parameters manually.".format(
                         args.pretrain_nranks, args.emb_dim, args.num_classes))
    else:
        with open(meta_file, 'r') as handle:
            config = json.load(handle)
        if args.pretrain_nranks < 0:
            args.pretrain_nranks = config['pretrain_nranks']
        elif args.pretrain_nranks != config['pretrain_nranks']:
            logger.error("The --pretrain_nranks ({}) parameter you set is not "
                         "equal to that ({}) for pretraining, please check "
                         "it.".format(args.pretrain_nranks, 
                             config['pretrain_nranks']))
            exit()
        if args.emb_dim < 0:
            args.emb_dim = config['emb_dim']
        elif args.emb_dim != config['emb_dim']:
            logger.error("The --emb_dim ({}) parameter you set is not equal to "
                         "that ({}) for pretraining, please check it.".format(
                             args.emb_dim, config['emb_dim']))
            exit()
        if args.num_classes < 0:
            args.num_classes = config['num_classes']
        elif args.num_classes != config['num_classes']:
            logger.error("The --num_classes ({}) parameter you set is not equal"
                         " to that ({}) for pretraining, please check "
                         "it.".format(args.emb_dim, config['emb_dim']))
            exit()
    logger.debug("Parameters for pretraining: pretrain_nranks ({}), emb_dim "
                 "({}), and num_classes ({}).".format(args.pretrain_nranks,
                     args.emb_dim, args.num_classes))
    logger.debug("Parameters for inference or finetuning: nranks ({}).".format(
                     args.nranks))


def find_distfc_var_names(args):
    """
    Find all names of pretrained distfc-related parameters, 
    e.g., dist_softmax_rank_00000.w_0, dist_softmax_rank_00000.b_0 etc.
    We assume that names of distfc-related parameters start with the 
    prefix 'dist'.
    """
    var_names = []
    model_dir = os.path.abspath(args.pretrained_model_dir)
    if not os.path.exists(model_dir):
        logger.error("The directory for pretrained model ({}) does not exist, "
                     "please check it.".format(model_dir))
        exit()
    logger.info("The directory for pretrained model: {}".format(model_dir))
    args.pretrained_model_dir = model_dir
    for file in os.listdir(model_dir):
        if args.name_feature in file:
            var_names.append(file)
    assert len(var_names) > 0, \
        logger.error("No distributed fc parameters found.")
    logger.info("Number of distributed fc parameters: {}.".format(
        len(var_names)))
    logger.debug("Distributed fc parameters: {}.".format(var_names))
    return var_names


def split_load_and_save(args, 
                        name_index,
                        param_names,
                        save_rank_id,
                        remainder,
                        as_bias,
                        train_nshards,
                        train_nranks,
                        nshards,
                        dtype="float32"):
    var2 = None
    advance = False
    emb_dim = args.emb_dim
    main_program = fluid.Program()
    startup_program = fluid.Program()

    load_var_name = param_names[name_index]
    save_var_name_list = load_var_name.split('.')
    save_var_name_list[0] = save_var_name_list[0].split('@')
    save_var_name_list[0][-1] = "%05d" % save_rank_id
    save_var_name_list[0] = '@'.join(save_var_name_list[0])
    save_var_name = '.'.join(save_var_name_list)

    last_train_nshards = args.num_classes - (train_nranks - 1) * train_nshards

    with fluid.program_guard(main_program, startup_program):
        if name_index == train_nranks - 1:
            var_dim = last_train_nshards
        else:
            var_dim = train_nshards

        shape = [var_dim] if as_bias else [emb_dim, var_dim]
        var = fluid.layers.create_parameter(shape, dtype=dtype,
            name=load_var_name)

        if as_bias:
            var = fluid.layers.slice(var, axes=[0],
                starts=[var.shape[0] - remainder], ends=[var.shape[0]])
        else:
            var = fluid.layers.split(var, [var.shape[1] - remainder, remainder],
                dim=1)[1]

        save_var_dim = nshards
        if remainder < nshards:
            if name_index == train_nranks - 1:
                save_var_dim = remainder
            else:
                name_index += 1
                advance = True
                load_var_name = param_names[name_index]

                if name_index == train_nranks - 1:
                    var_dim = last_train_nshards
                else:
                    var_dim = train_nshards
                shape = [var_dim] if as_bias else [emb_dim, var_dim]
                var2 = fluid.layers.create_parameter(shape, dtype=dtype,
                    name=load_var_name)

                if remainder + var_dim < nshards:
                    # The last train rank
                    save_var_dim = remainder + var_dim
                else:
                    remainder = remainder + var_dim - nshards
        elif remainder == nshards:
            if name_index == train_nranks - 2:
                remainder = last_train_nshards
                advance = True
            elif name_index < train_nranks - 2:
                remainder = train_nshards
                advance = True
        else:
            remainder = remainder - nshards
        if var2 is not None:
            var = fluid.layers.concat([var, var2], axis=0 if as_bias else 1)

        shape = [save_var_dim] if as_bias else [emb_dim, save_var_dim]
        to_save_var = fluid.layers.create_parameter(shape, dtype=dtype,
            name=save_var_name + '_temp')
        if save_var_dim != nshards: # get last dim
            if as_bias:
                temp_var = fluid.layers.slice(var, axes=[0], 
                    starts=[var.shape[0] - save_var_dim], ends=[var.shape[0]])
            else:
                temp_var = fluid.layers.split(var,
                    [var.shape[1] - save_var_dim, save_var_dim], dim=1)[1]
            fluid.layers.assign(temp_var, to_save_var)
        else:
            if as_bias:
                temp_var = fluid.layers.slice(var, axes=[0], starts=[0],
                    ends=[nshards])
            else:
                temp_var = fluid.layers.split(var,
                    [nshards, var.shape[1] - nshards], dim=1)[0]
            fluid.layers.assign(temp_var, to_save_var)

    def expected_var(var):
        has_var = os.path.exists(os.path.join(args.pretrained_model_dir,
            var.name))
        if has_var:
            return True
        return False
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_program)
    fluid.io.load_vars(exe, dirname=args.pretrained_model_dir,
        predicate=expected_var, main_program=main_program)
    exe.run(main_program)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    fluid.io.save_vars(exe, args.output_dir, vars=[to_save_var],
        main_program=main_program)
    srcfile = os.path.join(args.output_dir, to_save_var.name)
    dstfile = os.path.join(args.output_dir, save_var_name)
    shutil.move(srcfile, dstfile)
    return remainder, advance


def split_parameters(args, param_names, as_bias):
    """
    Split parameters whose names are in param_names.
    Params:
        args: command line paramters
        param_names: list of names of parameters to split
        as_bias: whether parameters to split are as bias or not
    """
    num_classes = args.num_classes
    train_nranks = args.pretrain_nranks
    nranks = args.nranks

    train_nshards = (num_classes + train_nranks - 1) // train_nranks
    nshards = (num_classes + nranks - 1) // nranks # for inference of finetuning

    save_rank_id = 0
    remainder_var_dim = train_nshards # remainder dim that is not split in a var
    name_index = 0 # index of name of pretrained parameter to process
    for save_rank_id in range(nranks):
        assert name_index < train_nranks
        remainder_var_dim, advance = split_load_and_save(args, name_index,
            param_names, save_rank_id, remainder_var_dim, as_bias,
            train_nshards, train_nranks, nshards)
        name_index += 1 if advance else 0
    processed_var_count = name_index + 1

    assert processed_var_count == train_nranks, logger.error("Number of "
        "pretrained parameters processed ({}) is not equal to the number of "
        "ranks ({}) for pretraining.".format(processed_var_count, train_nranks))
    assert save_rank_id == nranks - 1, logger.error("Number of saved parameters"
        " ({}) is not equal to the number of ranks ({}) for inference or "
        "finetuning.".format(save_rank_id + 1, nranks))


def split_distfc_parameters(args,
                            weight_param_names,
                            weight_velocity_param_names,
                            bias_param_names,
                            bias_velocity_param_names):
    """
    Split each distributed fc-related parameter according to number of ranks
    for inference or finetuning.
    
    Params:
        args: command line paramters
        weight_param_names: list of names of weight parameters
        bias_param_names: list of names of bias parameters
    """
    split_parameters(args, weight_param_names, as_bias=False)
    split_parameters(args, weight_velocity_param_names, as_bias=False)
    if len(bias_param_names) != 0:
        split_parameters(args, bias_param_names, as_bias=True)
        split_parameters(args, bias_velocity_param_names, as_bias=True)
    

def concat_load_and_save(args, 
                        name_index,
                        param_names,
                        save_rank_id,
                        remainder,
                        as_bias,
                        train_nshards,
                        train_nranks,
                        nshards,
                        dtype="float32"):
    advance = 0
    orig_nshards = nshards
    emb_dim = args.emb_dim
    main_program = fluid.Program()
    startup_program = fluid.Program()

    load_var_name = param_names[name_index]
    save_var_name_list = load_var_name.split('.')
    save_var_name_list[0] = save_var_name_list[0].split('@')
    save_var_name_list[0][-1] = "%05d" % save_rank_id
    save_var_name_list[0] = '@'.join(save_var_name_list[0])
    save_var_name = '.'.join(save_var_name_list)

    last_train_nshards = args.num_classes - (train_nranks - 1) * train_nshards

    with fluid.program_guard(main_program, startup_program):
        if name_index == train_nranks - 1:
            var_dim = last_train_nshards
        else:
            var_dim = train_nshards

        shape = [var_dim] if as_bias else [emb_dim, var_dim]
        var = fluid.layers.create_parameter(shape, dtype=dtype,
            name=load_var_name)

        if as_bias:
            var = fluid.layers.slice(var, axes=[0],
                starts=[var.shape[0] - remainder], ends=[var.shape[0]])
        else:
            var = fluid.layers.split(var, [var.shape[1] - remainder, remainder],
                dim=1)[1]
        to_concat_var_list = [var]
        while remainder < nshards and name_index < train_nranks - 1:
            name_index += 1
            advance += 1
            load_var_name = param_names[name_index]
            if name_index == train_nranks - 1:
                var_dim = last_train_nshards
            else:
                var_dim = train_nshards
            shape = [var_dim] if as_bias else [emb_dim, var_dim]
            var = fluid.layers.create_parameter(shape, dtype=dtype,
                name=load_var_name)

            to_concat_var_list.append(var)
            remainder += var_dim
        if len(to_concat_var_list) > 1:
            var = fluid.layers.concat(
                to_concat_var_list, axis=0 if as_bias else 1)
        save_var_dim = nshards
        if remainder > nshards:
            if as_bias:
                var = fluid.layers.slice(var, axes=[0], starts=[0],
                    ends=[nshards])
            else:
                var = fluid.layers.split(var,
                    [nshards, var.shape[1] - nshards], dim=1)[0]
            remainder = remainder - nshards
        elif remainder == nshards:
            if name_index == train_nranks - 2:
                #advance += 1 if len(to_concat_var_list) > 1 else 0 # to avoid duplicate add
                #name_index += 1 if len(to_concat_var_list) > 1 else 0 # to avoid duplicate add
                advance += 1
                name_index += 1
                remainder = last_train_nshards
            elif name_index < train_nranks - 2:
                #advance += 1 if len(to_concat_var_list) > 1 else 0 # to avoid duplicate add
                #name_index += 1 if len(to_concat_var_list) > 1 else 0 # to avoid duplicate add
                advance += 1
                name_index += 1
                remainder = train_nshards
        else:
            save_var_dim = remainder

        shape = [save_var_dim] if as_bias else [emb_dim, save_var_dim]
        to_save_var = fluid.layers.create_parameter(shape, dtype=dtype,
            name=save_var_name + '_temp')

        fluid.layers.assign(var, to_save_var)

    def expected_var(var):
        has_var = os.path.exists(os.path.join(args.pretrained_model_dir,
            var.name))
        if has_var:
            return True
        return False
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_program)
    fluid.io.load_vars(exe, dirname=args.pretrained_model_dir,
        predicate=expected_var, main_program=main_program)
    exe.run(main_program)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    fluid.io.save_vars(exe, args.output_dir, vars=[to_save_var],
        main_program=main_program)
    srcfile = os.path.join(args.output_dir, to_save_var.name)
    dstfile = os.path.join(args.output_dir, save_var_name)
    shutil.move(srcfile, dstfile)
    return remainder, advance


def concat_parameters(args, param_names, as_bias):
    """
    Concat parameters whose names are in param_names.
    Params:
        args: command line paramters
        param_names: list of names of parameters to concat
        as_bias: whether parameters to split are as bias or not
    """
    num_classes = args.num_classes
    train_nranks = args.pretrain_nranks
    nranks = args.nranks

    train_nshards = (num_classes + train_nranks - 1) // train_nranks
    nshards = (num_classes + nranks - 1) // nranks # for inference of finetuning

    save_rank_id = 0
    remainder_dim = train_nshards # remainder dim that is not concatted
    name_index = 0 # index of name of pretrained parameter to process
    for save_rank_id in range(nranks):
        assert name_index < train_nranks
        remainder_dim, advance = concat_load_and_save(args,
            name_index, param_names, save_rank_id, remainder_dim,
            as_bias, train_nshards, train_nranks, nshards)
        name_index += advance
    processed_var_count = name_index + 1

    assert processed_var_count == train_nranks, logger.error("Number of "
        "pretrained parameters processed ({}) is not equal to the number of "
        "ranks ({}) for pretraining.".format(processed_var_count, train_nranks))
    assert save_rank_id == nranks - 1, logger.error("Number of saved parameters"
        " ({}) is not equal to the number of ranks ({}) for inference or "
        "finetuning.".format(save_rank_id + 1, nranks))


def concat_distfc_parameters(args,
                             weight_param_names,
                             weight_velocity_param_names,
                             bias_param_names,
                             bias_velocity_param_names):
    """
    Concat distributed fc-related parameters according to number of ranks
    for inference or finetuning.
    
    Params:
        args: command line paramters
        weight_param_names: list of names of weight parameters
        bias_param_names: list of names of bias parameters
    """
    concat_parameters(args, weight_param_names, as_bias=False)
    concat_parameters(args, weight_velocity_param_names, as_bias=False)
    if len(bias_param_names) != 0:
        concat_parameters(args, bias_param_names, as_bias=True)
        concat_parameters(args, bias_velocity_param_names, as_bias=True)


def parameter_name_compare(x, y):
    """
    Compare two parameter names depend on their rank id.
    A parameter name is like dist_softmax_rank_00000.w_0,
    where 00000 is the rank id.
    """
    rank_id_x = int(x.split('.')[0].split('@')[-1])
    rank_id_y = int(y.split('.')[0].split('@')[-1])
    if rank_id_x < rank_id_y:
        return -1
    elif rank_id_x == rank_id_y:
        return 0
    else:
        return 1


def main():
    global args
    load_config(args)

    var_names = find_distfc_var_names(args)
    weight_param_names = [name for name in var_names 
        if '.w' in name and 'velocity' not in name]
    weight_velocity_param_names = [name for name in var_names 
        if '.w' in name and 'velocity' in name]
    bias_param_names = [name for name in var_names 
        if '.b' in name and 'velocity' not in name]
    bias_velocity_param_names = [name for name in var_names 
        if '.b' in name and 'velocity' in name]

    weight_param_names.sort(parameter_name_compare)
    weight_velocity_param_names.sort(parameter_name_compare)
    bias_param_names.sort(parameter_name_compare)
    bias_velocity_param_names.sort(parameter_name_compare)
    assert len(weight_param_names) == args.pretrain_nranks, \
        logger.error("Number of distributed fc-related weight parameters ({}) "
                     "should be equal to the number of ranks ({}) for "
                     "pretraining.".format(len(weight_param_names),
                         args.pretrain_nranks))
    assert len(weight_velocity_param_names) == args.pretrain_nranks, \
        logger.error("Number of distributed fc-related weight parameters ({}) "
                     "should be equal to the number of ranks ({}) for "
                     "pretraining.".format(len(weight_velocity_param_names),
                         args.pretrain_nranks))
    assert len(bias_param_names) == 0 or \
        len(bias_param_names) == args.pretrain_nranks, logger.error("Number of "
            "distributed fc-related bias parameters ({}) should be 0 or equal "
            "to the number of ranks ({}) for pretraining.".format(
                len(bias_param_names), args.pretrain_nranks))
    assert len(bias_velocity_param_names) == 0 or \
        len(bias_velocity_param_names) == args.pretrain_nranks, logger.error("Number of "
            "distributed fc-related bias parameters ({}) should be 0 or equal "
            "to the number of ranks ({}) for pretraining.".format(
                len(bias_velocity_param_names), args.pretrain_nranks))

    pretrain_nranks = args.pretrain_nranks
    nranks = args.nranks
    if pretrain_nranks == nranks:
        logger.info("Pre-training and inference (or finetuning) have the same "
                    "number of ranks, nothing to do.")
    elif pretrain_nranks < nranks:
        split_distfc_parameters(args, weight_param_names,
            weight_velocity_param_names, bias_param_names,
            bias_velocity_param_names)
    else:
        concat_distfc_parameters(args, weight_param_names,
            weight_velocity_param_names, bias_param_names,
            bias_velocity_param_names)

    logger.info("Done.")


if __name__ == "__main__":
    main()
