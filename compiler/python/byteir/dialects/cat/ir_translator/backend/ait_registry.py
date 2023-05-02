# Copyright 2022 ByteDance Ltd. and/or its affiliates. All rights reserved.
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
# ==============================================================================

import random
import numpy as np

from aitemplate.compiler.base import Tensor, _NumpyConstantTensorData
import aitemplate.compiler.ops as ait_ops
from aitemplate.compiler.ops import FuncEnum as ait_func_enum

from ..translator import IRTranslator
from byteir import ir
from byteir.utils import mlir_attr_to_pyobj

class AITemplateIRTranslator(IRTranslator):
    pass

@AITemplateIRTranslator.register("mhlo.constant")
def _dispatch_mhlo_constant(op, inputs):
    shaped_type = ir.ShapedType(op.result.type)
    shape = shaped_type.shape
    output = Tensor(shape)
    # TODO: need to support FP16
    #dtype = mlir_type_to_dtype(shaped_type.element_type)
    #np_array = mlir_attr_to_pyobj(op.attributes["value"])
    #output._bind_data(_NumpyConstantTensorData(np_array))
    return [output]

@AITemplateIRTranslator.register("cat.nchw2nhwc")
def _dispatch_cat_nchw2nhwc(op, inputs):
    ait_op = ait_ops.permute()
    Y = ait_op(inputs[0], [0, 2, 3, 1])
    return [Y]

@AITemplateIRTranslator.register("cat.conv2d")
def _dispatch_cat_conv2d(op, inputs):
    # NOTE: currently ait does not support normal conv2d with odd channel size
    # TODO: check layout, lhs_dilation and rhs_dilation here.
    stride = mlir_attr_to_pyobj(op.attributes["stride"])[0]
    pad = mlir_attr_to_pyobj(op.attributes["padding"])[0][0]
    dilation = mlir_attr_to_pyobj(op.attributes["rhs_dilation"])[0]
    ait_op = ait_ops.conv2d(stride=stride, pad=pad, dilate=dilation)
    Y = ait_op(inputs[0], inputs[1])
    return [Y]

@AITemplateIRTranslator.register("cat.conv2d_bias")
def _dispatch_cat_conv2d_bias(op, inputs):
    # NOTE: currently ait does not support normal conv2d with odd channel size
    # TODO: check layout, lhs_dilation and rhs_dilation here.
    stride = mlir_attr_to_pyobj(op.attributes["stride"])[0]
    pad = mlir_attr_to_pyobj(op.attributes["padding"])[0][0]
    dilation = mlir_attr_to_pyobj(op.attributes["rhs_dilation"])[0]

    shape = inputs[0].shape()
    in_channels = shape[3].value()
    if (in_channels <= 4):
        print("using conv_bias with few channels")
        inputs[1] = ait_ops.padding.nhwc3to4()(inputs[1])
        ait_op = ait_ops.conv2d_bias_few_channels(stride=stride, pad=pad, dilate=dilation)
    else:
        ait_op = ait_ops.conv2d_bias(stride=stride, pad=pad, dilate=dilation)

    Y = ait_op(inputs[0], inputs[1], inputs[2])
    return [Y]

@AITemplateIRTranslator.register("cat.conv2d_bias_relu")
def _dispatch_cat_conv2d_bias_relu(op, inputs):
    stride = mlir_attr_to_pyobj(op.attributes["stride"])[0]
    pad = mlir_attr_to_pyobj(op.attributes["padding"])[0][0]
    dilation = mlir_attr_to_pyobj(op.attributes["rhs_dilation"])[0]

    shape = inputs[0].shape()
    in_channels = shape[3].value()
    if (in_channels <= 4):
        print("using conv_bias with few channels")
        inputs[1] = ait_ops.padding.nhwc3to4()(inputs[1])
        ait_op = ait_ops.conv2d_bias_few_channels(stride=stride, pad=pad, dilate=dilation)
    else:
        ait_op = ait_ops.conv2d_bias_relu(stride=stride, pad=pad, dilate=dilation)
    Y = ait_op(inputs[0], inputs[1], inputs[2])
    return [Y]

@AITemplateIRTranslator.register("cat.batch_norm")
def _dispatch_cat_batch_norm(op, inputs):
    # TODO: AITemplate does not have BN impl
    raise NotImplementedError()

@AITemplateIRTranslator.register("cat.batch_matmul")
def _dispatch_cat_batch_norm(op, inputs):
    layout = mlir_attr_to_pyobj(op.attributes["layout"])
    if (layout == "ccr"):
        ait_op = ait_ops.bmm_ccr()
    elif (layout == "rrr"):
        ait_op = ait_ops.bmm_rrr()
    elif (layout == "crr"):
        ait_op = ait_ops.bmm_crr()
    elif (layout == "rcr"):
        ait_op = ait_ops.bmm_rcr()
    elif (layout == "ccc"):
        ait_op = ait_ops.bmm_ccr()
    elif (layout == "rrc"):
        ait_op = ait_ops.bmm_rrr()
    elif (layout == "crc"):
        ait_op = ait_ops.bmm_crr()
    elif (layout == "rcc"):
        ait_op = ait_ops.bmm_rcr()
    Y = ait_op(inputs[0], inputs[1])
    return [Y]

@AITemplateIRTranslator.register("cat.pooling2d")
def _dispatch_cat_pooling2d(op, inputs):
    window_stride = mlir_attr_to_pyobj(op.attributes["window_stride"])
    padding = mlir_attr_to_pyobj(op.attributes["padding"])
    kernel_size = mlir_attr_to_pyobj(op.attributes["kernel_size"])
    reduce_func = mlir_attr_to_pyobj(op.attributes["reduce_func"])
    if reduce_func == "max2d":
        ait_op = ait_ops.max_pool2d(kernel_size=kernel_size, stride=window_stride, pad=padding)
        Y = ait_op(inputs[0])
        return [Y]
    elif reduce_func == "avg2d":
        ait_op = ait_ops.avg_pool2d(kernel_size=kernel_size, stride=window_stride, pad=padding)
        Y = ait_op(inputs[0])
        return [Y]
    else:
        raise ValueError("Currently AIT backend only support max2d and avg2d pooling")

@AITemplateIRTranslator.register("cat.relu")
def _dispatch_cat_relu(op, inputs):
    Y = ait_ops.relu(inputs[0])
    return [Y]

@AITemplateIRTranslator.register("cat.reduce")
def _dispatch_cat_reduce(op, inputs):
    dim = mlir_attr_to_pyobj(op.attributes["dims"])
    dim = dim.tolist()

    # TODO: Currently aitemplate only support signle axis reduce
    if len(dim) == 1:
        ait_op = ait_ops.reduce_sum(dim=dim)
        Y = ait_op(inputs[0])
        return [Y]
    elif dim == [1, 2]: # TODO: use NHWC avg pooling but we need sum here
        tensor : Tensor = inputs[0]
        H, W = tensor.shape()[1:3]
        if H.value() != W.value():
            raise RuntimeError("Cannot handle pooling shape with H != W")
        ait_op = ait_ops.avg_pool2d(kernel_size=H.value(), stride=1, pad=0)
        Y_keepdim = ait_op(inputs[0])
        reshape_op = ait_ops.flatten(1)
        Y = reshape_op(Y_keepdim)
        return [Y]
    else:
        raise RuntimeError("Currently we do not support this kind of reduce") 

@AITemplateIRTranslator.register("cat.gemm")
def _dispatch_cat_gemm(op, inputs):
    layout = mlir_attr_to_pyobj(op.attributes["layout"])
    if layout == "rrr":
        ait_op = ait_ops.gemm_rrr()
        Y = ait_op(inputs[0], inputs[1])
        return [Y]
    elif layout == "rcr":
        ait_op = ait_ops.gemm_rcr()
        Y = ait_op(inputs[0], inputs[1])
        return [Y]
    else:
        raise RuntimeError("unsupported gemm layout")

@AITemplateIRTranslator.register("cat.gemm_bias")
def _dispatch_cat_gemm_bias(op, inputs):
    layout = mlir_attr_to_pyobj(op.attributes["layout"])
    if layout == "rrr":
        ait_op = ait_ops.gemm_rrr_bias()
        Y = ait_op(inputs[0], inputs[1], inputs[2])
        return [Y]
    elif layout == "rcr":
        ait_op = ait_ops.gemm_rcr_bias()
        Y = ait_op(inputs[0], inputs[1], inputs[2])
        return [Y]
    else:
        raise RuntimeError("unsupported gemm_bias layout")

@AITemplateIRTranslator.register("cat.bmm_permute")
def _dispatch_cat_bmm_permute(op, inputs):
    layout = mlir_attr_to_pyobj(op.attributes["layout"])
    d1 = mlir_attr_to_pyobj(op.attributes["shape"])
    if layout == "rrr":
        ait_op = ait_ops.bmm_rrr_permute(shape=(d1,))
        Y = ait_op(inputs[0], inputs[1])
        return [Y]
    elif layout == "rcr":
        ait_op = ait_ops.bmm_rcr_permute(shape=(d1,))
        Y = ait_op(inputs[0], inputs[1])
        return [Y]
    else:
        raise RuntimeError("unsupported bmm_permute layout")

@AITemplateIRTranslator.register("cat.softmax")
def _dispatch_cat_softmax(op, inputs):
    ait_op = ait_ops.softmax()
    dim = mlir_attr_to_pyobj(op.attributes["dim"])
    Y = ait_op(inputs[0], dim=dim)
    return [Y]

def _cat_elem_op_type_to_ait(op_type: str) -> ait_func_enum:
    op_map = {
        "add": ait_func_enum.ADD,
        "sub": ait_func_enum.SUB,
        "mul": ait_func_enum.MUL,
        "div": ait_func_enum.DIV,
        "tanh": ait_func_enum.TANH,
    }
    return op_map.get(op_type, None)

@AITemplateIRTranslator.register("cat.unary_elementwise")
def _dispatch_cat_unary_elementwise(op, inputs):
    op_type = mlir_attr_to_pyobj(op.attributes["op_type"])
    ait_op_type = _cat_elem_op_type_to_ait(op_type)
    if ait_op_type is None:
        raise ValueError(f"Invalid op type: {op_type}")
    ait_op = ait_ops.elementwise(ait_op_type)
    Y = ait_op(inputs[0])
    return [Y]

@AITemplateIRTranslator.register("cat.binary_elementwise")
def _dispatch_cat_binary_elementwise(op, inputs):
    op_type = mlir_attr_to_pyobj(op.attributes["op_type"])
    ait_op_type = _cat_elem_op_type_to_ait(op_type)
    if ait_op_type is None:
        raise ValueError(f"Invalid op type: {op_type}")
    ait_op = ait_ops.elementwise(ait_op_type)
    Y = ait_op(inputs[0], inputs[1])
    return [Y]

@AITemplateIRTranslator.register("mhlo.reshape")
def _dispatch_mhlo_reshape(op, inputs):
    shaped_type = ir.ShapedType(op.result.type)
    from aitemplate.compiler.ops.common.view_ops import reshape
    Y = reshape()(inputs[0], shaped_type.shape)
    return [Y]
