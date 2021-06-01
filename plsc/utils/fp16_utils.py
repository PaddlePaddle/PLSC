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

from paddle.fluid import core
from paddle.fluid.layer_helper import LayerHelper


def _rename_arg(op, old_name, new_name):
    """
    If an op has old_name input and output, rename these input
    args new_name.

    Args:
        op (Operator): Current operator.
        old_name (str): The old name of input args.
        new_name (str): The new name of input args.
    """
    op_desc = op.desc
    if isinstance(op_desc, tuple):
        op_desc = op_desc[0]
    op_desc._rename_input(old_name, new_name)
    op_desc._rename_output(old_name, new_name)


def _dtype_to_str(dtype):
    """
    Convert specific variable type to its corresponding string.

    Args:
        dtype (VarType): Variable type.
    """
    if dtype == core.VarDesc.VarType.FP16:
        return 'fp16'
    else:
        return 'fp32'


def _insert_cast_op(block, op, idx, src_dtype, dest_dtype):
    """
    Insert cast op and rename args of input and output.

    Args:
        block (Program): The block in which the operator is.
        op (Operator): The operator to insert cast op.
        idx (int): The index of current operator.
        src_dtype (VarType): The input variable dtype of cast op.
        dest_dtype (VarType): The output variable dtype of cast op.

    Returns:
        num_cast_op (int): The number of cast ops that have been inserted.
    """
    num_cast_ops = 0
    valid_types = [
        core.VarDesc.VarType.LOD_TENSOR, core.VarDesc.VarType.SELECTED_ROWS,
        core.VarDesc.VarType.LOD_TENSOR_ARRAY
    ]

    for in_name in op.input_names:
        if src_dtype == core.VarDesc.VarType.FP32 and op.type == 'batch_norm':
            if in_name != 'X':
                continue
        for in_var_name in op.input(in_name):
            in_var = block.var(in_var_name)
            if in_var.type not in valid_types:
                continue
            if in_var.dtype == src_dtype:
                cast_name = in_var.name + '.cast_' + _dtype_to_str(dest_dtype)
                out_var = block.vars.get(cast_name)
                if out_var is None or out_var.dtype != dest_dtype:
                    out_var = block.create_var(
                        name=cast_name,
                        dtype=dest_dtype,
                        persistable=False,
                        stop_gradient=False)

                    block._insert_op(
                        idx,
                        type="cast",
                        inputs={"X": in_var},
                        outputs={"Out": out_var},
                        attrs={
                            "in_dtype": in_var.dtype,
                            "out_dtype": out_var.dtype
                        })
                    num_cast_ops += 1
                _rename_arg(op, in_var.name, out_var.name)
            else:
                if op.has_attr('in_dtype'):
                    op._set_attr('in_dtype', dest_dtype)
    if src_dtype == core.VarDesc.VarType.FP32:
        for out_name in op.output_names:
            if op.type == 'batch_norm' and out_name != 'Y':
                continue
            for out_var_name in op.output(out_name):
                out_var = block.var(out_var_name)
                if out_var.type not in valid_types:
                    continue
                if out_var.dtype == core.VarDesc.VarType.FP32:
                    out_var.desc.set_dtype(core.VarDesc.VarType.FP16)
                    if op.has_attr('out_dtype'):
                        op._set_attr('out_dtype', core.VarDesc.VarType.FP16)
    return num_cast_ops


def check_op_validation(ops, idx, cur_op):
    """
    Check whether has op (backward_role) that inputs variable in
    cur_op's outputs behind cur_op.
    If yes, it means _move_optimize_ops_back will cause errors
    in program order. Therefore raise valueerror.
    If no, return True to continue moving in _move_optimize_ops_back.

    Args:
        ops (list): A list of ops.
        idx (int): index of cur_op in ops.
        cur_op (Operator): Current operator.
    """
    OPTIMIZE = core.op_proto_and_checker_maker.OpRole.Optimize
    for i in range(idx + 1, len(ops)):
        op = ops[i]
        if not op.attr('op_role') & int(OPTIMIZE):
            for input_name in op.input_arg_names:
                if input_name in cur_op.output_arg_names:
                    raise ValueError(
                        "There must be no next op that inputs {0} "
                        "variable after {1} op (optimize_role)".format(
                            var_name, cur_op.type))
    return True


def move_optimize_ops_back(block):
    """
    put optimize_role ops(cast) behind the backward_role ops
    to speed up in Executor.

    Args:
        block: The global_block of main program. eg: block = main_prog.global_block()
    """
    OPTIMIZE = core.op_proto_and_checker_maker.OpRole.Optimize
    optimize_ops = []
    for idx, op in reversed(list(enumerate(block.ops))):
        if op.attr('op_role') & int(OPTIMIZE):
            if check_op_validation(block.ops, idx, op):
                optimize_ops.append([
                    op.type, dict(
                        zip(op.input_names,
                            [block.var(name) for name in op.input_arg_names])),
                    dict(
                        zip(op.output_names, [
                            block.var(name) for name in op.output_arg_names
                        ])), op.all_attrs()
                ])
                block._remove_op(idx)
    for op in reversed(optimize_ops):
        assert len(op) == 4
        block.append_op(type=op[0], inputs=op[1], outputs=op[2], attrs=op[3])


def find_true_prev_op(ops, cur_op, var_name):
    """
    Find the true prev op that outputs var_name variable.

    Args:
        ops (list): A list of ops.
        cur_op (Operator): Current operator which has var_name variable.
        var_name (string): Variable name.
    """
    prev_op = []
    for op in ops:
        if op == cur_op:
            break
        for out_name in op.output_names:
            for out_var_name in op.output(out_name):
                if out_var_name == var_name:
                    prev_op.append(op)
    if prev_op:
        if not len(prev_op) == 1:
            raise ValueError("There must be only one previous op "
                             "that outputs {0} variable".format(var_name))
        else:
            return prev_op[0]
    return None


def _is_in_black_varnames(op, amp_lists):
    for in_name in op.input_arg_names:
        if in_name in amp_lists.black_varnames:
            return True

    for out_name in op.output_arg_names:
        if out_name in amp_lists.black_varnames:
            return True

    return False


def rewrite_program(main_prog, amp_lists):
    """
    Traverse all ops in current block and insert cast op according to
    which set current op belongs to.

    1. When an op belongs to the black list, add it to black set
    2. When an op belongs to the white list, add it to white set
    3. When an op belongs to the gray list. If one
       of its inputs is the output of black set op or black list op,
       add it to black set. If all of its previous ops are not black
       op and one of its inputs is the output of white set op or
       white list op, add it to white set.
    4. When an op isn't in the lists, add it to black op set.
    5. Add necessary cast ops to make sure that black set op will be
       computed in fp32 mode, while white set op will be computed in
       fp16 mode.

    Args:
        main_prog (Program): The main program for training.
    """
    block = main_prog.global_block()
    ops = block.ops
    white_op_set = set()
    black_op_set = set()
    for op in ops:
        if amp_lists.black_varnames is not None and _is_in_black_varnames(
                op, amp_lists):
            black_op_set.add(op)
            continue

        if op.type in amp_lists.black_list:
            black_op_set.add(op)
        elif op.type in amp_lists.white_list:
            white_op_set.add(op)
        elif op.type in amp_lists.gray_list:
            is_black_op = False
            is_white_op = False
            for in_name in op.input_names:
                # if this op has inputs
                if in_name:
                    for in_var_name in op.input(in_name):
                        in_var = block.var(in_var_name)
                        # this in_var isn't the output of other op
                        if in_var.op is None:
                            continue
                        elif in_var.op is op:
                            prev_op = find_true_prev_op(ops, op, in_var_name)
                            if prev_op is None:
                                continue
                        else:
                            prev_op = in_var.op
                        # if it's one of inputs
                        if prev_op in black_op_set or \
                                prev_op.type in amp_lists.black_list:
                            is_black_op = True
                        elif prev_op in white_op_set or \
                                prev_op.type in amp_lists.white_list:
                            is_white_op = True
            if is_black_op:
                black_op_set.add(op)
            elif is_white_op:
                white_op_set.add(op)
            else:
                pass
        else:
            # For numerical safe, we apply fp32 computation on ops that
            # are not determined which list they should stay.
            black_op_set.add(op)

    idx = 0
    while idx < len(ops):
        op = ops[idx]
        num_cast_ops = 0
        if op in black_op_set:
            num_cast_ops = _insert_cast_op(block, op, idx,
                                           core.VarDesc.VarType.FP16,
                                           core.VarDesc.VarType.FP32)
        elif op in white_op_set:
            num_cast_ops = _insert_cast_op(block, op, idx,
                                           core.VarDesc.VarType.FP32,
                                           core.VarDesc.VarType.FP16)
        else:
            pass

        idx += num_cast_ops + 1


def update_role_var_grad(main_prog, params_grads):
    """
    Update op_role_var attr for some ops to make sure the gradients
    transferred across GPUs is FP16.
    1. Check whether the op that outputs gradient is cast or not.
    2. If op is cast and gradient is FP32, remove the op_role_var
       and find the prev op which outputs FP16 gradient
    3. Update the op_role_var of the prev op.

    Args:
        main_prog (Program): The main program for training.
        params_grads (list): A list of params and grads.
    """
    block = main_prog.global_block()
    BACKWARD = core.op_proto_and_checker_maker.OpRole.Backward
    OPTIMIZE = core.op_proto_and_checker_maker.OpRole.Optimize
    for p, g in params_grads:
        op = g.op
        if g.dtype == core.VarDesc.VarType.FP32 and op.type == 'cast':
            role = op.attr('op_role')
            if role & int(BACKWARD) and op.has_attr('op_role_var'):
                op.desc.remove_attr("op_role_var")
            else:
                raise ValueError("The cast op {0} must be in BACKWARD role "
                                 "and have op_role_var attr.".format(op))

            fp16_grad_name = op.input(op.input_names[0])[0]
            op_for_fp16_grad = find_true_prev_op(block.ops, op, fp16_grad_name)
            op_role_var_attr_name = \
                core.op_proto_and_checker_maker.kOpRoleVarAttrName()
            attr_val = [p.name, fp16_grad_name]
            if op_for_fp16_grad.has_attr(op_role_var_attr_name):
                attr_val.extend(op_for_fp16_grad.attr(op_role_var_attr_name))
            op_for_fp16_grad._set_attr(op_role_var_attr_name, attr_val)

            # Maximize the all_reduce overlap, and perform the cast
            # operation after gradients transfer.
            op._set_attr('op_role', OPTIMIZE)

    move_optimize_ops_back(block)


def check_finite_and_unscale(x, scale, name=None):
    """
    Check if input X contains all finite data, if yes, scale it by input Scale.

    $$Out = X / scale$$

    If any tensor in X contains Inf or Nan, the Out will generate a indicator.
    FoundInfinite will be 1 (True), and Out will not be scaled. In this case, the data of 
    Out should not be used, and its data may not be deterministic. 
    Otherwise, FoundInfinite will be 0 (False).
    Args:
        x(list|tuple): The input tensors of check_finite_and_unscale operator.
        scale: The scale of check_finite_and_unscale operator.
    """
    #check_type(x, 'x', (tuple, list), 'check_finite_and_unscale')
    #for e in x:
    #    check_variable_and_dtype(e, "x", ['float16', 'float32', 'float64'],
    #                             'check_finite_and_unscale')

    helper = LayerHelper("check_finite_and_unscale", **locals())
    found_inf = helper.create_variable_for_type_inference(dtype='bool')

    inputs = {'X': x, 'Scale': scale}
    outputs = {'Out': x, 'FoundInfinite': found_inf}
    helper.append_op(
        type='check_finite_and_unscale', inputs=inputs, outputs=outputs)

    return x, found_inf


def update_loss_scaling(x,
                        found_inf,
                        prev_loss_scaling,
                        num_good_steps,
                        num_bad_steps,
                        incr_every_n_steps,
                        decr_every_n_nan_or_inf,
                        incr_ratio,
                        decr_ratio,
                        stop_update=False,
                        name=None):
    """
    Update loss scaling according to overall gradients. If all gradients is 
    finite after incr_every_n_steps, loss scaling will increase by incr_ratio. 
    Otherwise, loss scaling will decrease by decr_ratio after
    decr_every_n_nan_or_inf steps and each step some gradients are infinite.

    Args:
        x(list|tuple): The input tensors of update_loss_scaling operator.
        found_inf (Variable): A boolean variable indicates whether 
                                     there is any infinite gradient.
        prev_loss_scaling (Variable): Previous loss scaling.
        num_good_steps (Variable): A variable accumulates good steps in which 
                                   all gradients are finite.
        num_bad_steps (Variable): A variable accumulates bad steps in which 
                                  some gradients are infinite.
        incr_every_n_steps (int): A variable represents increasing loss 
                                       scaling every n consecutive steps with 
                                       finite gradients.
        decr_every_n_nan_or_inf (int): A variable represents decreasing 
                                            loss scaling every n accumulated 
                                            steps with nan or inf gradients.
        incr_ratio(float): The multiplier to use when increasing the loss 
                           scaling.
        decr_ratio(float): The less-than-one-multiplier to use when decreasing 
                           loss scaling.
    """

    #check_variable_and_dtype(prev_loss_scaling, "prev_loss_scaling",
    #                         ['float32', 'float64'], "update_loss_scaling")
    #check_type(x, 'x', (tuple, list), 'update_loss_scaling')
    for e in x:
        #check_variable_and_dtype(e, "x", ['float16', 'float32', 'float64'],
        #                         'update_loss_scaling')
        if e.dtype == core.VarDesc.VarType.FP16:
            assert prev_loss_scaling.dtype == core.VarDesc.VarType.FP32, \
                "The dtype of prev_loss_scaling should be float32 when the dtype of x is float16."
        else:
            assert prev_loss_scaling.dtype == e.dtype, "The dtype of prev_loss_scaling should be equal to the dtype of x."

    helper = LayerHelper("update_loss_scaling", **locals())

    inputs = {
        'X': x,
        'FoundInfinite': found_inf,
        'PrevLossScaling': prev_loss_scaling,
        'InGoodSteps': num_good_steps,
        'InBadSteps': num_bad_steps
    }

    outputs = {
        'Out': x,
        'LossScaling': prev_loss_scaling,
        'OutGoodSteps': num_good_steps,
        'OutBadSteps': num_bad_steps
    }

    attrs = {
        'incr_every_n_steps': incr_every_n_steps,
        'decr_every_n_nan_or_inf': decr_every_n_nan_or_inf,
        'incr_ratio': incr_ratio,
        'decr_ratio': decr_ratio,
        'stop_update': stop_update
    }

    helper.append_op(
        type='update_loss_scaling',
        inputs=inputs,
        outputs=outputs,
        attrs=attrs)

    return x
