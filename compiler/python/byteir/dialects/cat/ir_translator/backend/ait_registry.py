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
from byteir.utils import mlir_attr_to_pyobj, mlir_type_to_torch_str

class AITemplateIRTranslator(IRTranslator):
    pass

@AITemplateIRTranslator.register("mhlo.constant")
def _dispatch_mhlo_constant(op, inputs):
    shaped_type = ir.ShapedType(op.result.type)
    shape = shaped_type.shape
    output = Tensor(shape, dtype=mlir_type_to_torch_str(shaped_type.element_type))
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

@AITemplateIRTranslator.register("cat.reduce")
def _dispatch_cat_reduce(op, inputs):
    dim = mlir_attr_to_pyobj(op.attributes["dims"])
    reduce_ty = mlir_attr_to_pyobj(op.attributes["reduce_type"])
    dim = dim.tolist()

    # TODO: Currently aitemplate only support single axis reduce
    if len(dim) == 1:
        if reduce_ty == "sum":
            ait_op = ait_ops.reduce_sum(dim=dim)
            Y = ait_op(inputs[0])
            return [Y]
        else:
            raise RuntimeError("Currently we do not support 1D mean reduce") 
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

@AITemplateIRTranslator.register("cat.gemm_rrr")
def _dispatch_cat_gemm_rrr(op, inputs):
    ait_op = ait_ops.gemm_rrr()
    Y = ait_op(inputs[0], inputs[1])
    return [Y]

@AITemplateIRTranslator.register("cat.gemm_rcr")
def _dispatch_cat_gemm_rcr(op, inputs):
    ait_op = ait_ops.gemm_rcr()
    Y = ait_op(inputs[0], inputs[1])
    return [Y]

@AITemplateIRTranslator.register("cat.gemm_rrr_bias")
def _dispatch_cat_gemm_rrr_bias(op, inputs):
    ait_op = ait_ops.gemm_rrr_bias()
    Y = ait_op(inputs[0], inputs[1], inputs[2])
    return [Y]

@AITemplateIRTranslator.register("cat.gemm_rcr_bias")
def _dispatch_cat_gemm_rcr_bias(op, inputs):
    ait_op = ait_ops.gemm_rcr_bias()
    Y = ait_op(inputs[0], inputs[1], inputs[2])
    return [Y]

@AITemplateIRTranslator.register("cat.gemm_rcr_bias_relu")
def _dispatch_cat_gemm_rcr_bias_relu(op, inputs):
    ait_op = ait_ops.gemm_rcr_bias_relu()
    Y = ait_op(inputs[0], inputs[1], inputs[2])
    return [Y]

@AITemplateIRTranslator.register("cat.gemm_rcr_permute")
def _dispatch_cat_gemm_rcr_permute(op, inputs):
    t1 = mlir_attr_to_pyobj(op.attributes["t1"])
    t2 = mlir_attr_to_pyobj(op.attributes["t2"])
    ait_op = ait_ops.gemm_rcr_permute(shape=([t1, t2]), layout="0213")
    Y = ait_op(inputs[0], inputs[1])
    return [Y]

@AITemplateIRTranslator.register("cat.bmm_rrr_add")
def _dispatch_cat_bmm_rrr_add(op, inputs):
    ait_op = ait_ops.bmm_rrr_add()
    Y = ait_op(inputs[0], inputs[1], inputs[2])
    return [Y]

@AITemplateIRTranslator.register("cat.bmm_rrc_add")
def _dispatch_cat_bmm_rrc_add(op, inputs):
    ait_op = ait_ops.bmm_rrc_add()
    Y = ait_op(inputs[0], inputs[1], inputs[2])
    return [Y]

@AITemplateIRTranslator.register("cat.bmm_rcr_add")
def _dispatch_cat_bmm_rcr_add(op, inputs):
    ait_op = ait_ops.bmm_rcr_add()
    Y = ait_op(inputs[0], inputs[1], inputs[2])
    return [Y]

@AITemplateIRTranslator.register("cat.bmm_rcc_add")
def _dispatch_cat_bmm_rcc_add(op, inputs):
    ait_op = ait_ops.bmm_rcc_add()
    Y = ait_op(inputs[0], inputs[1], inputs[2])
    return [Y]

@AITemplateIRTranslator.register("cat.bmm_crr_add")
def _dispatch_cat_bmm_crr_add(op, inputs):
    ait_op = ait_ops.bmm_crr_add()
    Y = ait_op(inputs[0], inputs[1], inputs[2])
    return [Y]

@AITemplateIRTranslator.register("cat.bmm_crc_add")
def _dispatch_cat_bmm_crc_add(op, inputs):
    ait_op = ait_ops.bmm_crc_add()
    Y = ait_op(inputs[0], inputs[1], inputs[2])
    return [Y]

@AITemplateIRTranslator.register("cat.bmm_ccr_add")
def _dispatch_cat_bmm_ccr_add(op, inputs):
    ait_op = ait_ops.bmm_ccr_add()
    Y = ait_op(inputs[0], inputs[1], inputs[2])
    return [Y]

@AITemplateIRTranslator.register("cat.bmm_ccc_add")
def _dispatch_cat_bmm_ccc_add(op, inputs):
    ait_op = ait_ops.bmm_ccc_add()
    Y = ait_op(inputs[0], inputs[1], inputs[2])
    return [Y]

@AITemplateIRTranslator.register("cat.bmm_rrr_permute")
def _dispatch_cat_bmm_rrr_permute(op, inputs):
    d1 = mlir_attr_to_pyobj(op.attributes["shape"])
    ait_op = ait_ops.bmm_rrr_permute(shape=(d1,))
    Y = ait_op(inputs[0], inputs[1])
    return [Y]

@AITemplateIRTranslator.register("cat.bmm_rcr_permute")
def _dispatch_cat_bmm_rcr_permute(op, inputs):
    d1 = mlir_attr_to_pyobj(op.attributes["shape"])
    ait_op = ait_ops.bmm_rcr_permute(shape=(d1,))
    Y = ait_op(inputs[0], inputs[1])
    return [Y]

@AITemplateIRTranslator.register("cat.bmm_rrr")
def _dispatch_cat_bmm_rrr(op, inputs):
    ait_op = ait_ops.bmm_rrr()
    Y = ait_op(inputs[0], inputs[1])
    return [Y]

@AITemplateIRTranslator.register("cat.bmm_rrc")
def _dispatch_cat_bmm_rrc(op, inputs):
    ait_op = ait_ops.bmm_rrc()
    Y = ait_op(inputs[0], inputs[1])
    return [Y]

@AITemplateIRTranslator.register("cat.bmm_rcr")
def _dispatch_cat_bmm_rcr(op, inputs):
    ait_op = ait_ops.bmm_rcr()
    Y = ait_op(inputs[0], inputs[1])
    return [Y]

@AITemplateIRTranslator.register("cat.bmm_rcc")
def _dispatch_cat_bmm_rcc(op, inputs):
    ait_op = ait_ops.bmm_rcc()
    Y = ait_op(inputs[0], inputs[1])
    return [Y]

@AITemplateIRTranslator.register("cat.bmm_crr")
def _dispatch_cat_bmm_crr(op, inputs):
    ait_op = ait_ops.bmm_crr()
    Y = ait_op(inputs[0], inputs[1])
    return [Y]

@AITemplateIRTranslator.register("cat.bmm_crc")
def _dispatch_cat_bmm_crc(op, inputs):
    ait_op = ait_ops.bmm_crc()
    Y = ait_op(inputs[0], inputs[1])
    return [Y]

@AITemplateIRTranslator.register("cat.bmm_ccr")
def _dispatch_cat_bmm_ccr(op, inputs):
    ait_op = ait_ops.bmm_ccr()
    Y = ait_op(inputs[0], inputs[1])
    return [Y]

@AITemplateIRTranslator.register("cat.bmm_ccc")
def _dispatch_cat_bmm_ccc(op, inputs):
    ait_op = ait_ops.bmm_ccc()
    Y = ait_op(inputs[0], inputs[1])
    return [Y]

@AITemplateIRTranslator.register("cat.softmax")
def _dispatch_cat_softmax(op, inputs):
    ait_op = ait_ops.softmax()
    dim = mlir_attr_to_pyobj(op.attributes["dim"])
    Y = ait_op(inputs[0], dim=dim)
    return [Y]

@AITemplateIRTranslator.register("cat.layernorm")
def _dispatch_cat_layernorm(op, inputs):
    ait_op = ait_ops.layernorm()
    axises = mlir_attr_to_pyobj(op.attributes["axis"])
    eps = mlir_attr_to_pyobj(op.attributes["epsilon"])
    shape = inputs[0].shape()
    normalized_shape = []
    for axis in axises:
        normalized_shape.append(shape[axis])
    Y = ait_op(inputs[0], inputs[1], inputs[2], normalized_shape, eps)
    return [Y]

@AITemplateIRTranslator.register("mhlo.transpose")
def _dispatch_mhlo_transpose(op, inputs):
    shaped_type = ir.ShapedType(op.result.type)
    dims = mlir_attr_to_pyobj(op.attributes["permutation"])
    dims = dims.tolist()
    Y = ait_ops.permute()(inputs[0], dims)
    return [Y]
