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