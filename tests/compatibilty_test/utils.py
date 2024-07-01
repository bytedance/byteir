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

import torch

def mlir_type_to_torch_dtype(mlir_type):
    if str(mlir_type) in ["DType.float64"]:
        return torch.float64
    if str(mlir_type) in ["DType.float32"]:
        return torch.float32
    if str(mlir_type) in ["DType.float16"]:
        return torch.float16
    if str(mlir_type) in ["DType.int64", "DType.index"]:
        return torch.int64
    if str(mlir_type) in ["DType.int32"]:
        return torch.int32
    if str(mlir_type) in ["DType.int16"]:
        return torch.int16
    if str(mlir_type) in ["DType.int8"]:
        return torch.int8
    if str(mlir_type) in ["DType.int1", "DType.bool"]:
        return torch.bool
    raise NotImplementedError("unsupported mlir type {}".format(
        str(mlir_type)))
