import os
import sys
import time
import argparse
import functools
import numpy as np

import paddle
import paddle.fluid as fluid
import resnet
import sklearn
import reader
from verification import evaluate
from utility import add_arguments, print_arguments
from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy
from paddle.fluid.incubate.fleet.collective import DistFCConfig
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.transpiler.details.program_utils import program_to_code
from paddle.fluid.optimizer import Optimizer
import paddle.fluid.profiler as profiler
from fp16_utils import rewrite_program, update_role_var_grad, update_loss_scaling, move_optimize_ops_back
from fp16_lists import AutoMixedPrecisionLists
from paddle.fluid.transpiler.details import program_to_code
import paddle.fluid.layers as layers
import paddle.fluid.unique_name as unique_name

parser = argparse.ArgumentParser(description="Train parallel face network.")
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('train_batch_size', int,   128,         "Minibatch size for training.")
add_arg('test_batch_size',  int,   120,         "Minibatch size for test.")
add_arg('num_epochs',       int,   120,         "Number of epochs to run.")
add_arg('image_shape',      str,   "3,112,112", "Image size in the format of CHW.")
add_arg('emb_dim',          int,   512,         "Embedding dim size.")
add_arg('class_dim',        int,   85742,       "Number of classes.")
add_arg('model_save_dir',   str,   None,        "Directory to save model.")
add_arg('pretrained_model', str,   None,        "Directory for pretrained model.")
add_arg('lr',               float, 0.1,         "Initial learning rate.")
add_arg('model',            str,   "ResNet_ARCFACE50", "The network to use.")
add_arg('loss_type',        str,   "softmax",   "Type of network loss to use.")
add_arg('margin',           float, 0.5,         "Parameter of margin for arcface or dist_arcface.")
add_arg('scale',            float, 64.0,        "Parameter of scale for arcface or dist_arcface.")
add_arg('with_test',        bool, False,        "Whether to do test during training.")
add_arg('fp16',             bool, True,        "Whether to do test during training.")
add_arg('profile',          bool,  False,                "Enable profiler or not." )
# yapf: enable
args = parser.parse_args()


model_list = [m for m in dir(resnet) if "__" not in m]



def optimizer_setting(params, args):
    ls = params["learning_strategy"]
    step = 1
    bd = [step * e for e in ls["epochs"]]
    base_lr = params["lr"]
    lr = [base_lr * (0.1 ** i) for i in range(len(bd) + 1)]
    print("bd: {}".format(bd))
    print("lr_step: {}".format(lr))
    step_lr = fluid.layers.piecewise_decay(boundaries=bd, values=lr)
    optimizer = fluid.optimizer.Momentum(
        learning_rate=step_lr,
        momentum=0.9,
        regularization=fluid.regularizer.L2Decay(5e-4))
    num_trainers = int(os.getenv("PADDLE_TRAINERS_NUM", 1))
    if args.loss_type in ["dist_softmax", "dist_arcface"]:
        if args.fp16:
            wrapper = DistributedClassificationOptimizer(
                optimizer, args.train_batch_size * num_trainers, step_lr, 
                loss_type=args.loss_type, init_loss_scaling=1.0)
        else:
            wrapper = DistributedClassificationOptimizer(optimizer, args.train_batch_size * num_trainers, step_lr)
    elif args.loss_type in ["softmax", "arcface"]:
        wrapper = optimizer


    return wrapper


def build_program(args,
                  main_program,
                  startup_program,
                  is_train=True,
                  use_parallel_test=False,
                  fleet=None,
                  strategy=None):
    model_name = args.model
    assert model_name in model_list, \
        "{} is not in supported lists: {}".format(args.model, model_list)
    assert not (is_train and use_parallel_test), \
        "is_train and use_parallel_test cannot be set simultaneously"

    trainer_id = int(os.getenv("PADDLE_TRAINER_ID", 0))
    worker_num = int(os.getenv("PADDLE_TRAINERS_NUM", 1))

    image_shape = [int(m) for m in args.image_shape.split(",")]
    # model definition
    model = resnet.__dict__[model_name]()
    with fluid.program_guard(main_program, startup_program):
        with fluid.unique_name.guard():
            image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')
            emb, loss = model.net(input=image,
                                  label=label,
                                  is_train=is_train,
                                  emb_dim=args.emb_dim,
                                  class_dim=args.class_dim,
                                  loss_type=args.loss_type,
                                  margin=args.margin,
                                  scale=args.scale)
            if args.loss_type in ["dist_softmax", "dist_arcface"]:
                shard_prob = loss._get_info("shard_prob")
                prob_all = fluid.layers.collective._c_allgather(shard_prob,
                    nranks=worker_num, use_calc_stream=True)
                prob_list = fluid.layers.split(prob_all, dim=0,
                    num_or_sections=worker_num)
                prob = fluid.layers.concat(prob_list, axis=1)
                label_all = fluid.layers.collective._c_allgather(label,
                    nranks=worker_num, use_calc_stream=True)
                acc1 = fluid.layers.accuracy(input=prob, label=label_all, k=1)
                acc5 = fluid.layers.accuracy(input=prob, label=label_all, k=5)
            elif args.loss_type in ["softmax", "arcface"]:
                prob = loss[1]
                loss = loss[0]
                acc1 = fluid.layers.accuracy(input=prob, label=label, k=1)
                acc5 = fluid.layers.accuracy(input=prob, label=label, k=5)
            optimizer = None
            if is_train:
                # parameters from model and arguments
                params = model.params
                params["lr"] = args.lr
                params["num_epochs"] = args.num_epochs
                params["learning_strategy"]["batch_size"] = args.train_batch_size
                # initialize optimizer
                optimizer = optimizer_setting(params, args)
                dist_optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
                dist_optimizer.minimize(loss)
            elif use_parallel_test:
                emb = fluid.layers.collective._c_allgather(emb,
                    nranks=worker_num, use_calc_stream=True)
    return emb, loss, acc1, acc5, optimizer


def train(args):
    pretrained_model = args.pretrained_model
    model_save_dir = args.model_save_dir
    model_name = args.model

    trainer_id = int(os.getenv("PADDLE_TRAINER_ID", 0))
    worker_num = int(os.getenv("PADDLE_TRAINERS_NUM", 1))

    role = role_maker.PaddleCloudRoleMaker(is_collective=True)
    fleet.init(role)
    strategy = DistributedStrategy()
    strategy.mode = "collective"
    strategy.collective_mode = "grad_allreduce"

    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    test_program = fluid.Program()
    train_emb, train_loss, train_acc1, train_acc5, optimizer = \
        build_program(args, train_prog, startup_prog, True, False,
                      fleet, strategy)
    test_emb, test_loss, test_acc1, test_acc5, _ = \
        build_program(args, test_program, startup_prog, False, True)

    if args.loss_type in ["dist_softmax", "dist_arcface"]:
        if not args.fp16:
            global_lr = optimizer._optimizer._global_learning_rate(
                program=train_prog)
        else:
            global_lr = optimizer._optimizer._global_learning_rate(
                program=train_prog)
    elif args.loss_type in ["softmax", "arcface"]:
        global_lr = optimizer._global_learning_rate(program=train_prog)

    origin_prog = fleet._origin_program
    train_prog = fleet.main_program
    if trainer_id == 0:
        with open('start.program', 'w') as fout:
            program_to_code(startup_prog, fout, True)
        with open('main.program', 'w') as fout:
            program_to_code(train_prog, fout, True)
        with open('origin.program', 'w') as fout:
            program_to_code(origin_prog, fout, True)

    gpu_id = int(os.getenv("FLAGS_selected_gpus", 0))
    place = fluid.CUDAPlace(gpu_id)
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    if pretrained_model:
        pretrained_model = os.path.join(pretrained_model, str(trainer_id))
        def if_exist(var):
            has_var = os.path.exists(os.path.join(pretrained_model, var.name))
            if has_var:
                print('var: %s found' % (var.name))
            return has_var
        fluid.io.load_vars(exe, pretrained_model, predicate=if_exist,
            main_program=train_prog)

    train_reader = paddle.batch(reader.arc_train(args.class_dim),
        batch_size=args.train_batch_size)
    if args.with_test:
        test_list, test_name_list = reader.test()
        test_feeder = fluid.DataFeeder(place=place, feed_list=['image', 'label'], program=test_program)
        fetch_list_test = [test_emb.name, test_acc1.name, test_acc5.name]
    feeder = fluid.DataFeeder(place=place, feed_list=['image', 'label'], program=train_prog)

    fetch_list_train = [train_loss.name, global_lr.name, train_acc1.name, train_acc5.name,train_emb.name,"loss_scaling_0"]
    # test_program = test_program._prune(targets=loss)

    num_trainers = int(os.getenv("PADDLE_TRAINERS_NUM", 1))
    real_batch_size = args.train_batch_size * num_trainers
    real_test_batch_size = args.test_batch_size * num_trainers
    local_time = 0.0
    nsamples = 0
    inspect_steps = 100
    step_cnt = 0
    for pass_id in range(args.num_epochs):
        train_info = [[], [], [], []]
        local_train_info = [[], [], [], []]
        for batch_id, data in enumerate(train_reader()):
            nsamples += real_batch_size
            t1 = time.time()
            loss, lr, acc1, acc5, train_embedding, loss_scaling = exe.run(train_prog, feed=feeder.feed(data),
                fetch_list=fetch_list_train, use_program_cache=True)
            t2 = time.time()
            if args.profile and step_cnt == 50:
                print("begin profiler")
                if trainer_id == 0:
                    profiler.start_profiler("All")
            elif args.profile and batch_id == 55:
                print("begin to end profiler")
                if trainer_id == 0:
                    profiler.stop_profiler("total", "./profile_%d" % (trainer_id))
                print("end profiler break!")
                args.profile=False


            period = t2 - t1
            local_time += period
            train_info[0].append(np.array(loss)[0])
            train_info[1].append(np.array(lr)[0])
            local_train_info[0].append(np.array(loss)[0])
            local_train_info[1].append(np.array(lr)[0])
            if batch_id % inspect_steps == 0:
                avg_loss = np.mean(local_train_info[0])
                avg_lr = np.mean(local_train_info[1])
                print("Pass:%d batch:%d lr:%f loss:%f qps:%.2f acc1:%.4f acc5:%.4f" % (
                    pass_id, batch_id, avg_lr, avg_loss, nsamples / local_time,
                    acc1, acc5))
                #print("train_embedding:,",np.array(train_embedding)[0])
                print("train_embedding is nan:",np.isnan(np.array(train_embedding)[0]).sum())
                print("loss_scaling",loss_scaling)
                local_time = 0
                nsamples = 0
                local_train_info = [[], [], [], []]
            step_cnt += 1

            if args.with_test and step_cnt % inspect_steps == 0:
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
                            offset = trainer_id * args.test_batch_size
                            begin = start + offset
                            end = begin + args.test_batch_size
                            _data = []
                            for k in xrange(begin, end):
                                _data.append((data[k], 0))
                            assert len(_data) == args.test_batch_size
                            [_embeddings, acc1, acc5] = exe.run(test_program,
                                fetch_list = fetch_list_test, feed=test_feeder.feed(_data),
                                use_program_cache=True)
                            if embeddings is None:
                                embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
                            embeddings[start:start+real_test_batch_size, :] = _embeddings[:, :]
                        beg = parallel_test_steps * real_test_batch_size

                        while beg < data.shape[0]:
                            end = min(beg + args.test_batch_size, data.shape[0])
                            count = end - beg
                            _data = []
                            for k in xrange(end - args.test_batch_size, end):
                                _data.append((data[k], 0))
                            [_embeddings, acc1, acc5] = exe.run(test_program,
                                fetch_list = fetch_list_test, feed=test_feeder.feed(_data),
                                use_program_cache=True)
                            _embeddings = _embeddings[0:args.test_batch_size,:]
                            embeddings[beg:end, :] = _embeddings[(args.test_batch_size-count):, :]
                            beg = end
                        embeddings_list.append(embeddings)

                    xnorm = 0.0
                    xnorm_cnt = 0
                    for embed in embeddings_list:
                        xnorm += np.sqrt((embed * embed).sum(axis=1)).sum(axis=0)
                        xnorm_cnt += embed.shape[0]
                    xnorm /= xnorm_cnt

                    embeddings = embeddings_list[0] + embeddings_list[1]
                    if np.isnan(embeddings).sum() > 1:
                        print("======test np.isnan(embeddings).sum()",np.isnan(embeddings).sum())
                        continue
                    embeddings = sklearn.preprocessing.normalize(embeddings)
                    _, _, accuracy, val, val_std, far = evaluate(embeddings, issame_list, nrof_folds=10)
                    acc, std = np.mean(accuracy), np.std(accuracy)

                    print('[%s][%d]XNorm: %f' % (test_name_list[i], step_cnt, xnorm))
                    print('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (test_name_list[i], step_cnt, acc, std))
                    sys.stdout.flush()
                test_end = time.time()
                print("test time: {}".format(test_end - test_start))

        train_loss = np.array(train_info[0]).mean()
        print("End pass {0}, train_loss {1}".format(pass_id, train_loss))
        sys.stdout.flush()

        #save model
        #if trainer_id == 0:
        if model_save_dir:
            model_path = os.path.join(model_save_dir + '/' + model_name,
                                  str(pass_id), str(trainer_id))
            if not os.path.isdir(model_path):
                os.makedirs(model_path)
            fluid.io.save_persistables(exe, model_path)


class DistributedClassificationOptimizer(Optimizer):
    '''
    A optimizer wrapper to generate backward network for distributed
    classification training of model parallelism.
    '''

    def __init__(self,optimizer, batch_size, lr, 
                 loss_type='dist_arcface',
                 amp_lists=None,
                 init_loss_scaling=1.0,
                 incr_every_n_steps=1000,
                 decr_every_n_nan_or_inf=2,
                 incr_ratio=2.0,
                 decr_ratio=0.5,
                 use_dynamic_loss_scaling=True):
        super(DistributedClassificationOptimizer, self).__init__(
            learning_rate=lr)
        self._optimizer = optimizer
        self._batch_size = batch_size
        self._amp_lists = amp_lists
        if amp_lists is None:
            self._amp_lists = AutoMixedPrecisionLists()

        self._param_grads = None
        self._scaled_loss = None
        self._loss_type = loss_type
        self._init_loss_scaling = init_loss_scaling
        self._loss_scaling = layers.create_global_var(
            name=unique_name.generate("loss_scaling"),
            shape=[1],
            value=init_loss_scaling,
            dtype='float32',
            persistable=True)
        self._use_dynamic_loss_scaling = use_dynamic_loss_scaling
        if self._use_dynamic_loss_scaling:
            self._incr_every_n_steps = layers.fill_constant(
                shape=[1], dtype='int32', value=incr_every_n_steps)
            self._decr_every_n_nan_or_inf = layers.fill_constant(
                shape=[1], dtype='int32', value=decr_every_n_nan_or_inf)
            self._incr_ratio = incr_ratio
            self._decr_ratio = decr_ratio
            self._num_good_steps = layers.create_global_var(
                name=unique_name.generate("num_good_steps"),
                shape=[1],
                value=0,
                dtype='int32',
                persistable=True)
            self._num_bad_steps = layers.create_global_var(
                name=unique_name.generate("num_bad_steps"),
                shape=[1],
                value=0,
                dtype='int32',
                persistable=True)

        # Ensure the data type of learning rate vars is float32 (same as the
        # master parameter dtype)
        if isinstance(optimizer._learning_rate, float):
            optimizer._learning_rate_map[fluid.default_main_program()] = \
                        layers.create_global_var(
                        name=unique_name.generate("learning_rate"),
                        shape=[1],
                        value=float(optimizer._learning_rate),
                        dtype='float32',
                        persistable=True)

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None,
                 callbacks=None):
        assert loss._get_info('shard_logit')

        shard_logit = loss._get_info('shard_logit')
        shard_prob = loss._get_info('shard_prob')
        shard_label = loss._get_info('shard_label')
        shard_dim = loss._get_info('shard_dim')

        op_maker = fluid.core.op_proto_and_checker_maker
        op_role_key = op_maker.kOpRoleAttrName()
        op_role_var_key = op_maker.kOpRoleVarAttrName()
        backward_role = int(op_maker.OpRole.Backward)
        loss_backward_role = int(op_maker.OpRole.Loss) | int(
            op_maker.OpRole.Backward)

        # minimize a scalar of reduce_sum to generate the backward network
        scalar = fluid.layers.reduce_sum(shard_logit)
        if not args.fp16:
            ret = self._optimizer.minimize(scalar)
            with open("fp32_before.program", "w") as f:
                program_to_code(block.program,fout=f, skip_op_callstack=False)

            block = loss.block
            # remove the unnecessary ops
            index = 0
            for i, op in enumerate(block.ops):
                if op.all_attrs()[op_role_key] == loss_backward_role:
                    index = i
                    break
            print("op_role_key: ",op_role_key)
            print("loss_backward_role:",loss_backward_role)
            # print("\nblock.ops: ",block.ops)
            print("block.ops[index - 1].type: ", block.ops[index - 1].type)
            print("block.ops[index].type: ", block.ops[index].type)
            print("block.ops[index + 1].type: ", block.ops[index + 1].type)

            assert block.ops[index - 1].type == 'reduce_sum'
            assert block.ops[index].type == 'fill_constant'
            assert block.ops[index + 1].type == 'reduce_sum_grad'
            block._remove_op(index + 1)
            block._remove_op(index)
            block._remove_op(index - 1)

            # insert the calculated gradient
            dtype = shard_logit.dtype
            shard_one_hot = fluid.layers.create_tensor(dtype, name='shard_one_hot')
            block._insert_op(
                index - 1,
                type='one_hot',
                inputs={'X': shard_label},
                outputs={'Out': shard_one_hot},
                attrs={
                    'depth': shard_dim,
                    'allow_out_of_range': True,
                    op_role_key: backward_role
                })
            shard_logit_grad = fluid.layers.create_tensor(
                dtype, name=fluid.backward._append_grad_suffix_(shard_logit.name))
            block._insert_op(
                index,
                type='elementwise_sub',
                inputs={'X': shard_prob,
                        'Y': shard_one_hot},
                outputs={'Out': shard_logit_grad},
                attrs={op_role_key: backward_role})
            block._insert_op(
                index + 1,
                type='scale',
                inputs={'X': shard_logit_grad},
                outputs={'Out': shard_logit_grad},
                attrs={
                    'scale': 1.0 / self._batch_size,
                    op_role_key: loss_backward_role
                })
            with open("fp32_after.program", "w") as f:
                program_to_code(block.program,fout=f, skip_op_callstack=False)

        # use mixed_precision for training
        else:
            block = loss.block
            rewrite_program(block.program, self._amp_lists)
            self._params_grads = self._optimizer.backward(
                scalar, startup_program, parameter_list, no_grad_set,
                callbacks)
            update_role_var_grad(block.program, self._params_grads)
            move_optimize_ops_back(block.program.global_block())
            scaled_params_grads = []
            for p, g in self._params_grads:
                with fluid.default_main_program()._optimized_guard([p, g]):
                    scaled_g = g / self._loss_scaling
                    scaled_params_grads.append([p, scaled_g])

            index = 0
            for i, op in enumerate(block.ops):
                if op.all_attrs()[op_role_key] == loss_backward_role:
                    index = i
                    break
            fp32 = fluid.core.VarDesc.VarType.FP32
            dtype = shard_logit.dtype

            if self._loss_type == 'dist_arcface':
                assert block.ops[index - 2].type == 'fill_constant'
                assert block.ops[index - 1].type == 'reduce_sum'
                assert block.ops[index].type == 'fill_constant'
                assert block.ops[index + 1].type == 'reduce_sum_grad'
                assert block.ops[index + 2].type == 'scale'
                assert block.ops[index + 3].type == 'elementwise_add_grad'

                block._remove_op(index + 2)
                block._remove_op(index + 1)
                block._remove_op(index)
                block._remove_op(index - 1)

                # insert the calculated gradient
                shard_one_hot = fluid.layers.create_tensor(dtype, name='shard_one_hot')
                block._insert_op(
                    index - 1,
                    type='one_hot',
                    inputs={'X': shard_label},
                    outputs={'Out': shard_one_hot},
                    attrs={
                        'depth': shard_dim,
                        'allow_out_of_range': True,
                        op_role_key: backward_role
                    })
                shard_one_hot_fp32 = fluid.layers.create_tensor(fp32, name=(shard_one_hot.name+".cast_fp32"))
                block._insert_op(
                    index,
                    type="cast",
                    inputs={"X": shard_one_hot},
                    outputs={"Out": shard_one_hot_fp32},
                    attrs={
                        "in_dtype": fluid.core.VarDesc.VarType.FP16,
                        "out_dtype": fluid.core.VarDesc.VarType.FP32,
                        op_role_key: backward_role
                    })
                name = 'tmp_3@GRAD'
                shard_logit_grad_fp32 = block.var(name)

                block._insert_op(
                    index+1,
                    type='elementwise_sub',
                    inputs={'X': shard_prob,
                            'Y': shard_one_hot_fp32},
                    outputs={'Out': shard_logit_grad_fp32},
                    attrs={op_role_key: backward_role})

                block._insert_op(
                    index+2,
                    type='elementwise_mul',
                    inputs={'X': shard_logit_grad_fp32,
                            'Y': self._loss_scaling},
                    outputs={'Out': shard_logit_grad_fp32},
                    attrs={op_role_key: backward_role})

                block._insert_op(
                    index+3,
                    type='scale',
                    inputs={'X': shard_logit_grad_fp32},
                    outputs={'Out': shard_logit_grad_fp32},
                    attrs={
                        'scale': 1.0 / self._batch_size,
                        op_role_key: loss_backward_role
                    })
            elif self._loss_type == 'dist_softmax':
                print("block.ops[index - 3].type: ", block.ops[index - 3].type)
                print("block.ops[index - 2].type: ", block.ops[index - 2].type)
                print("block.ops[index-1].type: ", block.ops[index - 1].type)
                print("block.ops[index].type: ", block.ops[index].type)
                print("block.ops[index + 1].type: ", block.ops[index +1].type)
                print("block.ops[index + 2].type: ", block.ops[index +2].type)
                print("block.ops[index + 3].type: ", block.ops[index +3].type)
                with open("fp16_softmax_before.program", "w") as f:
                    program_to_code(block.program,fout=f, skip_op_callstack=False)

                assert block.ops[index - 1].type == 'reduce_sum'
                assert block.ops[index].type == 'fill_constant'
                assert block.ops[index + 1].type == 'reduce_sum_grad'
                assert block.ops[index + 2].type == 'cast'
                assert block.ops[index + 3].type == 'elementwise_add_grad'
                
                block._remove_op(index + 1)
                block._remove_op(index)
                block._remove_op(index - 1)

                # insert the calculated gradient 
                shard_one_hot = fluid.layers.create_tensor(fp32, name='shard_one_hot')
                shard_one_hot_fp32 = fluid.layers.create_tensor(fp32, 
                    name=(shard_one_hot.name+".cast_fp32"))
                shard_logit_grad_fp32 = block.var(shard_logit.name+".cast_fp32@GRAD")
                block._insert_op(
                    index - 1,
                    type='one_hot',
                    inputs={'X': shard_label},
                    outputs={'Out': shard_one_hot_fp32},
                    attrs={
                        'depth': shard_dim,
                        'allow_out_of_range': True,
                        op_role_key: backward_role
                    })
                
                block._insert_op(
                    index,
                    type='elementwise_sub',
                    inputs={'X': shard_prob,
                            'Y': shard_one_hot_fp32},
                    outputs={'Out': shard_logit_grad_fp32},
                    attrs={op_role_key: backward_role})
                block._insert_op(
                    index + 1,
                    type='elementwise_mul',
                    inputs={'X': shard_logit_grad_fp32,
                            'Y': self._loss_scaling},
                    outputs={'Out': shard_logit_grad_fp32},
                    attrs={op_role_key: backward_role})
                block._insert_op(
                    index + 2,
                    type='scale',
                    inputs={'X': shard_logit_grad_fp32},
                    outputs={'Out': shard_logit_grad_fp32},
                    attrs={
                        'scale': 1.0 / self._batch_size,
                        op_role_key: loss_backward_role
                    })

            if self._use_dynamic_loss_scaling:
                grads = [layers.reduce_sum(g) for [_, g] in scaled_params_grads]
                all_grads = layers.concat(grads)
                all_grads_sum = layers.reduce_sum(all_grads)
                is_overall_finite = layers.isfinite(all_grads_sum)

                update_loss_scaling(is_overall_finite, self._loss_scaling,
                                    self._num_good_steps, self._num_bad_steps,
                                    self._incr_every_n_steps,
                                    self._decr_every_n_nan_or_inf, self._incr_ratio,
                                    self._decr_ratio)

                with layers.Switch() as switch:
                    with switch.case(is_overall_finite):
                        pass
                    with switch.default():
                        for _, g in scaled_params_grads:
                            layers.assign(layers.zeros_like(g), g)

            optimize_ops = self._optimizer.apply_gradients(scaled_params_grads)
            ret = optimize_ops, scaled_params_grads

            with open("fp16_softmax.program", "w") as f:
                program_to_code(block.program,fout=f, skip_op_callstack=False)
        return ret



def main():
    global args
    all_loss_types = ["softmax", "arcface", "dist_softmax", "dist_arcface"]
    assert args.loss_type in all_loss_types, \
        "All supported loss types [{}], but give {}.".format(
            all_loss_types, args.loss_type)
    print_arguments(args)
    train(args)


if __name__ == '__main__':
    main()
