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

import json
import logging
import os
import shutil
from functools import cmp_to_key

import paddle.fluid as fluid

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s %(asctime)s line:%(lineno)d] %(message)s',
    datefmt='%d %b %Y %H:%M:%S')
logger = logging.getLogger()


class ParameterConverter(object):
    """
    Tool to convert pre-trained distributed fc parameters for inference or
    fine-tuning. Note that the number of ranks or GPUs for inference or
    fine-tuning can be different from that for pre-training.
    """

    def __init__(self, model_dir, output_dir, num_trainers):
        super(ParameterConverter, self).__init__()
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.pretrain_nranks = -1
        self.emb_dim = -1
        self.num_classes = -1
        self.nranks = num_trainers

        self.load_config()

    def load_config(self):
        """
        Load config file which contains the following information for
        pre-training:
            1. pretrain_nranks (int): number of ranks for pre-training;
            2. emb_dim (int): embedding dim for pre-training;
            3. num_classes (int): number of classes for classification.
        """
        meta_file = os.path.join(self.model_dir, 'meta.json')
        if not os.path.exists(meta_file):
            logger.error("Meta file does not exist, make sure your pre-trained "
                         "models are legal.")
            exit()

        with open(meta_file, 'r') as handle:
            config = json.load(handle)

        self.pretrain_nranks = config['pretrain_nranks']
        assert self.pretrain_nranks > 0
        self.emb_dim = config['emb_dim']
        assert self.emb_dim > 0
        self.num_classes = config['num_classes']
        assert self.num_classes > 0

        logger.info("Parameters for pre-training: pretrain_nranks ({}), "
                    "emb_dim ({}), and num_classes ({}).".format(
            self.pretrain_nranks,
            self.emb_dim,
            self.num_classes))
        logger.debug("Parameters for inference or fine-tuning: "
                     "nranks ({}).".format(self.nranks))

    def find_var_names(self):
        """
        Find all names of pre-trained parameters for the distributed fc layer,
        e.g., dist@softmax@rank@00000.w_0, dist@softmax@rank@00000.b_0 etc.
        We assume that names of distributed fc related parameters start with the
        prefix dist@ and have @rank@ in their names.
        """
        var_names = []
        model_dir = os.path.abspath(self.model_dir)
        if not os.path.exists(model_dir):
            logger.error("The directory for pre-trained model ({}) does not "
                         "exist, please check it.".format(model_dir))
            exit()
        logger.info("The directory for pre-trained model: {}".format(model_dir))
        for file in os.listdir(model_dir):
            if 'dist@' in file and '@rank@' in file:
                var_names.append(file)
        assert len(var_names) > 0, \
            logger.error("No distributed fc parameters found.")
        logger.info("Number of distributed fc parameters: {}.".format(
            len(var_names)))
        logger.info("Distributed fc parameters: {}.".format(var_names))
        return var_names

    def split_load_and_save(self,
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
        emb_dim = self.emb_dim
        main_program = fluid.Program()
        startup_program = fluid.Program()
        num_classes = self.num_classes

        load_var_name = param_names[name_index]
        save_var_name_list = load_var_name.split('.')
        save_var_name_list[0] = save_var_name_list[0].split('@')
        save_var_name_list[0][-1] = "%05d" % save_rank_id
        save_var_name_list[0] = '@'.join(save_var_name_list[0])
        save_var_name = '.'.join(save_var_name_list)

        last_train_nshards = num_classes - (train_nranks - 1) * train_nshards

        with fluid.program_guard(main_program, startup_program):
            if name_index == train_nranks - 1:
                var_dim = last_train_nshards
            else:
                var_dim = train_nshards

            shape = [var_dim] if as_bias else [emb_dim, var_dim]
            var = fluid.layers.create_parameter(shape,
                                                dtype=dtype,
                                                name=load_var_name)

            if as_bias:
                var = fluid.layers.slice(var,
                                         axes=[0],
                                         starts=[var.shape[0] - remainder],
                                         ends=[var.shape[0]])
            else:
                var = fluid.layers.split(var,
                                         [var.shape[1] - remainder,
                                          remainder],
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
                    var2 = fluid.layers.create_parameter(shape,
                                                         dtype=dtype,
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
            to_save_var = fluid.layers.create_parameter(
                shape,
                dtype=dtype,
                name=save_var_name + '_temp')
            if save_var_dim != nshards:  # get last dim
                if as_bias:
                    temp_var = fluid.layers.slice(
                        var,
                        axes=[0],
                        starts=[var.shape[0] - save_var_dim],
                        ends=[var.shape[0]])
                else:
                    temp_var = fluid.layers.split(
                        var,
                        [var.shape[1] - save_var_dim, save_var_dim],
                        dim=1)[1]
                fluid.layers.assign(temp_var, to_save_var)
            else:
                if as_bias:
                    temp_var = fluid.layers.slice(var,
                                                  axes=[0],
                                                  starts=[0],
                                                  ends=[nshards])
                else:
                    temp_var = fluid.layers.split(
                        var,
                        [nshards, var.shape[1] - nshards],
                        dim=1)[0]
                fluid.layers.assign(temp_var, to_save_var)

        def expected_var(var):
            has_var = os.path.exists(os.path.join(self.model_dir, var.name))
            if has_var:
                return True
            return False

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)
        fluid.io.load_vars(exe,
                           dirname=self.model_dir,
                           predicate=expected_var,
                           main_program=main_program)
        exe.run(main_program)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        fluid.io.save_vars(exe,
                           self.output_dir,
                           vars=[to_save_var],
                           main_program=main_program)
        srcfile = os.path.join(self.output_dir, to_save_var.name)
        dstfile = os.path.join(self.output_dir, save_var_name)
        shutil.move(srcfile, dstfile)
        return remainder, advance

    def split_parameters(self, param_names, as_bias):
        """
        Split parameters whose names are in param_names.
        Params:
            param_names: list of names of parameters to split
            as_bias: whether parameters to split are as bias or not
        """
        num_classes = self.num_classes
        train_nranks = self.pretrain_nranks
        nranks = self.nranks

        train_nshards = (num_classes + train_nranks - 1) // train_nranks
        nshards = (num_classes + nranks - 1) // nranks

        save_rank_id = 0
        # remainder dim that is not split in a var
        remainder_var_dim = train_nshards
        name_index = 0  # index of name of pre-trained parameter to process
        for save_rank_id in range(nranks):
            assert name_index < train_nranks
            remainder_var_dim, advance = self.split_load_and_save(
                name_index,
                param_names,
                save_rank_id,
                remainder_var_dim,
                as_bias,
                train_nshards,
                train_nranks,
                nshards)
            name_index += 1 if advance else 0
        processed_var_count = name_index + 1

        assert processed_var_count == train_nranks, \
            logger.error("Number of pre-trained parameters processed ({}) is "
                         "not equal to the number of ranks ({}) for "
                         "pre-training.".format(processed_var_count,
                                                train_nranks))
        assert save_rank_id == nranks - 1, \
            logger.error("Number of saved parameters ({}) is not equal to the "
                         "number of ranks ({}) for inference or "
                         "fine-tuning.".format(save_rank_id + 1, nranks))

    def split_distfc_parameters(self,
                                weight_param_names,
                                weight_velocity_param_names,
                                bias_param_names,
                                bias_velocity_param_names):
        """
        Split each distributed fc-related parameter according to number of ranks
        for inference or fine-tuning.

        Params:
            weight_param_names: list of names of weight parameters
            bias_param_names: list of names of bias parameters
        """
        self.split_parameters(weight_param_names, as_bias=False)
        self.split_parameters(weight_velocity_param_names, as_bias=False)
        if len(bias_param_names) != 0:
            self.split_parameters(bias_param_names, as_bias=True)
            self.split_parameters(bias_velocity_param_names, as_bias=True)

    def concat_load_and_save(self,
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
        emb_dim = self.emb_dim
        main_program = fluid.Program()
        startup_program = fluid.Program()
        num_classes = self.num_classes

        load_var_name = param_names[name_index]
        save_var_name_list = load_var_name.split('.')
        save_var_name_list[0] = save_var_name_list[0].split('@')
        save_var_name_list[0][-1] = "%05d" % save_rank_id
        save_var_name_list[0] = '@'.join(save_var_name_list[0])
        save_var_name = '.'.join(save_var_name_list)

        last_train_nshards = num_classes - (train_nranks - 1) * train_nshards

        with fluid.program_guard(main_program, startup_program):
            if name_index == train_nranks - 1:
                var_dim = last_train_nshards
            else:
                var_dim = train_nshards

            shape = [var_dim] if as_bias else [emb_dim, var_dim]
            var = fluid.layers.create_parameter(shape,
                                                dtype=dtype,
                                                name=load_var_name)

            if as_bias:
                var = fluid.layers.slice(var,
                                         axes=[0],
                                         starts=[var.shape[0] - remainder],
                                         ends=[var.shape[0]])
            else:
                var = fluid.layers.split(var,
                                         [var.shape[1] - remainder,
                                          remainder],
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
                var = fluid.layers.create_parameter(shape,
                                                    dtype=dtype,
                                                    name=load_var_name)

                to_concat_var_list.append(var)
                remainder += var_dim
            if len(to_concat_var_list) > 1:
                var = fluid.layers.concat(to_concat_var_list,
                                          axis=0 if as_bias else 1)
            save_var_dim = nshards
            if remainder > nshards:
                if as_bias:
                    var = fluid.layers.slice(var,
                                             axes=[0],
                                             starts=[0],
                                             ends=[nshards])
                else:
                    var = fluid.layers.split(
                        var,
                        [nshards, var.shape[1] - nshards],
                        dim=1)[0]
                remainder = remainder - nshards
            elif remainder == nshards:
                if name_index == train_nranks - 2:
                    # advance += 1 if len(to_concat_var_list) > 1 else 0
                    # to avoid duplicate add
                    # name_index += 1 if len(to_concat_var_list) > 1 else 0
                    # to avoid duplicate add
                    advance += 1
                    name_index += 1
                    remainder = last_train_nshards
                elif name_index < train_nranks - 2:
                    # advance += 1 if len(to_concat_var_list) > 1 else 0
                    # to avoid duplicate add
                    # name_index += 1 if len(to_concat_var_list) > 1 else 0
                    # to avoid duplicate add
                    advance += 1
                    name_index += 1
                    remainder = train_nshards
            else:
                save_var_dim = remainder

            shape = [save_var_dim] if as_bias else [emb_dim, save_var_dim]
            to_save_var = fluid.layers.create_parameter(
                shape,
                dtype=dtype,
                name=save_var_name + '_temp')

            fluid.layers.assign(var, to_save_var)

        def expected_var(var):
            has_var = os.path.exists(os.path.join(self.model_dir, var.name))
            if has_var:
                return True
            return False

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)
        fluid.io.load_vars(exe,
                           dirname=self.model_dir,
                           predicate=expected_var,
                           main_program=main_program)
        exe.run(main_program)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        fluid.io.save_vars(exe,
                           self.output_dir,
                           vars=[to_save_var],
                           main_program=main_program)
        srcfile = os.path.join(self.output_dir, to_save_var.name)
        dstfile = os.path.join(self.output_dir, save_var_name)
        shutil.move(srcfile, dstfile)
        return remainder, advance

    def concat_parameters(self, param_names, as_bias):
        """
        Concat parameters whose names are in param_names.
        Params:
            param_names: list of names of parameters to concat
            as_bias: whether parameters to split are as bias or not
        """
        num_classes = self.num_classes
        train_nranks = self.pretrain_nranks
        nranks = self.nranks

        train_nshards = (num_classes + train_nranks - 1) // train_nranks
        nshards = (num_classes + nranks - 1) // nranks

        save_rank_id = 0
        remainder_dim = train_nshards  # remainder dim that is not concated
        name_index = 0  # index of name of pre-trained parameter to process
        for save_rank_id in range(nranks):
            assert name_index < train_nranks
            remainder_dim, advance = self.concat_load_and_save(name_index,
                                                               param_names,
                                                               save_rank_id,
                                                               remainder_dim,
                                                               as_bias,
                                                               train_nshards,
                                                               train_nranks,
                                                               nshards)
            name_index += advance
        processed_var_count = name_index + 1

        assert processed_var_count == train_nranks, \
            logger.error("Number of pre-trained parameters processed ({}) is "
                         "not equal to the number of ranks ({}) for "
                         "pre-training.".format(processed_var_count,
                                                train_nranks))
        assert save_rank_id == nranks - 1, \
            logger.error("Number of saved parameters ({}) is not equal to the "
                         "number of ranks ({}) for inference or "
                         "fine-tuning.".format(save_rank_id + 1, nranks))

    def concat_distfc_parameters(self,
                                 weight_param_names,
                                 weight_velocity_param_names,
                                 bias_param_names,
                                 bias_velocity_param_names):
        """
        Concat distributed fc-related parameters according to number of ranks
        for inference or finetuning.

        Params:
            weight_param_names: list of names of weight parameters
            weight_velocity_param_names: list of names of weight velocity
                parameters
            bias_param_names: list of names of bias parameters
            bias_velocity_param_names: list of names of bias velocity parameters
        """
        self.concat_parameters(weight_param_names, as_bias=False)
        self.concat_parameters(weight_velocity_param_names, as_bias=False)
        if len(bias_param_names) != 0:
            self.concat_parameters(bias_param_names, as_bias=True)
            self.concat_parameters(bias_velocity_param_names, as_bias=True)

    def process(self):
        self.load_config()
        var_names = self.find_var_names()
        weight_param_names = [name for name in var_names
                              if '.w' in name and 'velocity' not in name]
        weight_velocity_param_names = [name for name in var_names
                                       if '.w' in name and 'velocity' in name]
        bias_param_names = [name for name in var_names
                            if '.b' in name and 'velocity' not in name]
        bias_velocity_param_names = [name for name in var_names
                                     if '.b' in name and 'velocity' in name]

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

        weight_param_names.sort(key=cmp_to_key(parameter_name_compare))
        weight_velocity_param_names.sort(
            key=cmp_to_key(parameter_name_compare))
        bias_param_names.sort(key=cmp_to_key(parameter_name_compare))
        bias_velocity_param_names.sort(key=cmp_to_key(parameter_name_compare))

        assert len(weight_param_names) == self.pretrain_nranks, \
            logger.error(
                "Number of distributed fc-related weight parameters ({}) "
                "should be equal to the number of ranks ({}) for "
                "pre-training.".format(len(weight_param_names),
                                       self.pretrain_nranks))
        assert len(weight_velocity_param_names) == self.pretrain_nranks, \
            logger.error(
                "Number of distributed fc-related weight parameters ({}) "
                "should be equal to the number of ranks ({}) for "
                "pre-training.".format(len(weight_velocity_param_names),
                                       self.pretrain_nranks))
        assert (len(bias_param_names) == 0 or
                len(bias_param_names) == self.pretrain_nranks), \
            logger.error(
                "Number of distributed fc-related bias parameters ({}) "
                "should be 0 or equal to the number of ranks ({}) for "
                "pre-training.".format(len(bias_param_names),
                                       self.pretrain_nranks))
        assert (len(bias_velocity_param_names) == 0 or
                len(bias_velocity_param_names) == self.pretrain_nranks), \
            logger.error(
                "Number of distributed fc-related bias parameters ({}) "
                "should be 0 or equal to the number of ranks ({}) for "
                "pre-training.".format(len(bias_velocity_param_names),
                                       self.pretrain_nranks))

        pretrain_nranks = self.pretrain_nranks
        nranks = self.nranks
        if pretrain_nranks == nranks:
            logger.info(
                "Pre-training and inference (or fine-tuning) have the same "
                "number of ranks, nothing to do.")
        elif pretrain_nranks < nranks:
            self.split_distfc_parameters(weight_param_names,
                                         weight_velocity_param_names,
                                         bias_param_names,
                                         bias_velocity_param_names)
        else:
            self.concat_distfc_parameters(weight_param_names,
                                          weight_velocity_param_names,
                                          bias_param_names,
                                          bias_velocity_param_names)

        logger.info("Done.")


if __name__ == "__main__":
    converter = ParameterConverter('./trained_model',
                                   "./trained_model_temp",
                                   8)
    converter.process()
