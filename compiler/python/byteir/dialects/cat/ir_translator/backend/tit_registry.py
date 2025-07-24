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
from tritontemplate.compiler.base import Tensor
import tritontemplate.compiler.ops as tit_ops

from ..translator import IRTranslator
from byteir import ir
from byteir.utils import mlir_attr_to_pyobj, mlir_type_to_torch_str

class TRITONTemplateIRTranslator(IRTranslator):
    pass

@TRITONTemplateIRTranslator.register("mhlo.constant")
def _dispatch_mhlo_constant(op, inputs):
    shaped_type = ir.ShapedType(op.result.type)
    shape = shaped_type.shape
    output = Tensor(shape, dtype=mlir_type_to_torch_str(shaped_type.element_type))
    return [output]

@TRITONTemplateIRTranslator.register("cat.gemm_rcr_bias_relu")
def _dispatch_cat_gemm_rcr_bias_relu(op, inputs):
    Y = tit_ops.Gemm(inputs=inputs, layout="rcr", is_bias=True, activation="relu")
    return [Y]

@TRITONTemplateIRTranslator.register("cat.gemm_rcr_bias")
def _dispatch_cat_gemm_rcr_bias(op, inputs):
    Y = tit_ops.Gemm(inputs=inputs, layout="rcr", is_bias=True)
    return [Y]

@TRITONTemplateIRTranslator.register("cat.gemm_rcr_relu")
def _dispatch_cat_gemm_rcr_relu(op, inputs):
    Y = tit_ops.Gemm(inputs=inputs, layout="rcr", activation="relu")
    return [Y]

@TRITONTemplateIRTranslator.register("cat.gemm_rcr")
def _dispatch_cat_gemm_rcr(op, inputs):
    Y = tit_ops.Gemm(inputs=inputs, layout="rcr")
    return [Y]

@TRITONTemplateIRTranslator.register("cat.gemm_rrr_bias_relu")
def _dispatch_cat_gemm_rrr_bias_relu(op, inputs):
    Y = tit_ops.Gemm(inputs=inputs, layout="rrr", is_bias=True, activation="relu")
    return [Y]

@TRITONTemplateIRTranslator.register("cat.gemm_rrr_bias")
def _dispatch_cat_gemm_rrr_bias(op, inputs):
    Y = tit_ops.Gemm(inputs=inputs, layout="rrr", is_bias=True)
    return [Y]

@TRITONTemplateIRTranslator.register("cat.gemm_rrr_relu")
def _dispatch_cat_gemm_rrr_relu(op, inputs):
    Y = tit_ops.Gemm(inputs=inputs, layout="rrr", activation="relu")
    return [Y]

@TRITONTemplateIRTranslator.register("cat.gemm_rrr")
def _dispatch_cat_gemm_rrr(op, inputs):
    Y = tit_ops.Gemm(inputs=inputs, layout="rrr")
    return [Y]

@TRITONTemplateIRTranslator.register("cat.bmm_rrr")
def _dispatch_cat_bmm_rrr(op, inputs):
    Y = tit_ops.Bmm(inputs=inputs, layout="rrr")
    return [Y]

@TRITONTemplateIRTranslator.register("cat.bmm_rrr_add")
def _dispatch_cat_bmm_rrr_add(op, inputs):
    Y = tit_ops.Bmm(inputs=inputs, layout="rrr", is_bias=True)
    return [Y]

@TRITONTemplateIRTranslator.register("cat.bmm_rcr")
def _dispatch_cat_bmm_rcr(op, inputs):
    Y = tit_ops.Bmm(inputs=inputs, layout="rcr")
    return [Y]

@TRITONTemplateIRTranslator.register("cat.bmm_rcr_add")
def _dispatch_cat_bmm_rcr_add(op, inputs):
    Y = tit_ops.Bmm(inputs=inputs, layout="rcr", is_bias=True)
    return [Y]

@TRITONTemplateIRTranslator.register("cat.bmm_crr")
def _dispatch_cat_bmm_crr(op, inputs):
    Y = tit_ops.Bmm(inputs=inputs, layout="crr")
    return [Y]

@TRITONTemplateIRTranslator.register("cat.bmm_crr_add")
def _dispatch_cat_bmm_crr_add(op, inputs):
    Y = tit_ops.Bmm(inputs=inputs, layout="crr", is_bias=True)
    return [Y]

@TRITONTemplateIRTranslator.register("cat.bmm_ccr")
def _dispatch_cat_bmm_ccr(op, inputs):
    Y = tit_ops.Bmm(inputs=inputs, layout="ccr")
    return [Y]

@TRITONTemplateIRTranslator.register("cat.bmm_ccr_add")
def _dispatch_cat_bmm_ccr_add(op, inputs):
    Y = tit_ops.Bmm(inputs=inputs, layout="ccr", is_bias=True)
    return [Y]

@TRITONTemplateIRTranslator.register("cat.softmax")
def _dispatch_cat_softmax(op, inputs):
    shaped_type = ir.ShapedType(op.result.type)
    shape = shaped_type.shape
    dtype = mlir_type_to_torch_str(shaped_type.element_type)
    outputs=[Tensor(shape=shape,dtype=dtype,name='output_0')]
    dim = mlir_attr_to_pyobj(op.attributes["dim"])
    Y = tit_ops.Softmax(inputs=inputs,dim=dim,outputs=outputs,enable_online=True)
    return [Y]

@TRITONTemplateIRTranslator.register("cat.layernorm")
def _dispatch_cat_layernorm(op, inputs):
    axises = mlir_attr_to_pyobj(op.attributes["axis"])
    eps = mlir_attr_to_pyobj(op.attributes["epsilon"])
    Y=tit_ops.Layernorm(inputs=inputs,axises=axises,eps=eps)
    return [Y]

@TRITONTemplateIRTranslator.register("mhlo.transpose")
def _dispatch_mhlo_transpose(op, inputs):
    dims = mlir_attr_to_pyobj(op.attributes["permutation"])
    dims = dims.tolist()
    dims_str = ''.join(map(str, dims))
    Y=tit_ops.Transpose(inputs=inputs,permutation=dims_str)
    return [Y]
