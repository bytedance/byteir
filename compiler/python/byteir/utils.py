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

import numpy as np
from byteir import ir
import torch

def mlir_type_to_np_dtype(mlir_type):
    if str(mlir_type) == "f64":
        return np.float64
    if str(mlir_type) == "f32":
        return np.float32
    if str(mlir_type) == "f16":
        return np.half
    if str(mlir_type) == "i64":
        return np.int64
    if str(mlir_type) == "ui64":
        return np.uint64
    if str(mlir_type) == "i32":
        return np.int32
    if str(mlir_type) == "ui32":
        return np.uint32
    if str(mlir_type) == "i16":
        return np.int16
    if str(mlir_type) == "ui16":
        return np.uint16
    if str(mlir_type) == "i8":
        return np.int8
    if str(mlir_type) == "ui8":
        return np.uint8
    if str(mlir_type) == "i1":
        return np.bool_
    if str(mlir_type) == "!tf_type.string":
        return np.str_
    if str(mlir_type) == "!ace.string":
        return np.str_
    if str(mlir_type) == "index":
        return np.int64
    raise NotImplementedError("unsupported mlir type {}".format(mlir_type))


def mlir_attr_to_pyobj(attribute):
    if ir.DictAttr.isinstance(attribute):
        dict_attr = ir.DictAttr(attribute)
        return {
            dict_attr[idx].name: mlir_attr_to_pyobj(dict_attr[idx].attr)
            for idx in range(len(dict_attr))
        }

    if ir.ArrayAttr.isinstance(attribute):
        array_attr = ir.ArrayAttr(attribute)
        return [mlir_attr_to_pyobj(i) for i in array_attr]

    for DenseElementsAttrCls in [ir.DenseIntElementsAttr, ir.DenseFPElementsAttr]:
        if DenseElementsAttrCls.isinstance(attribute):
            dense_attr = DenseElementsAttrCls(attribute)
            assert ir.ShapedType.isinstance(dense_attr.type)
            dense_attr_type = ir.ShapedType(dense_attr.type)
            return np.array(
                [i for i in dense_attr],
                dtype=mlir_type_to_np_dtype(dense_attr_type.element_type),
            ).reshape(dense_attr_type.shape)

    for attr_type_name in [
        "StringAttr",
        "BoolAttr",
        "IntegerAttr",
        "FloatAttr",
        "FlatSymbolRefAttr",
    ]:
        attr_type_cls = getattr(ir, attr_type_name)
        if attr_type_cls.isinstance(attribute):
            return attr_type_cls(attribute).value

    raise NotImplementedError("unsupported attribute {}".format(attribute))

def mlir_type_to_torch_str(mlir_type) -> str:
    _map = {
        "bf16": "bfloat16",
        "f16": "float16",
        "f32": "float32",
        "f64": "float64",
        "i8": "int8",
        "i16": "int16",
        "i32": "int32",
        "i64": "int64",
        "i1": "bool",
    }
    return _map.get(str(mlir_type), None)

def torch_dtype_from_str(dtype_name: str) -> torch.dtype:
    _map = {
        "float": torch.float,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
        "int": torch.int,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "bool": torch.bool,
    }
    return _map.get(dtype_name, None)
