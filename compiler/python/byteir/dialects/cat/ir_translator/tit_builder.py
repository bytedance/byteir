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
import torch
import os
import numpy as np
from typing import Set

from tritontemplate.compiler.base import Tensor, IntImm
from tritontemplate.compiler import compile_kernel

from .backend.tit_registry import *

from byteir.utils import mlir_type_to_torch_str, torch_dtype_from_str

class TITBuilder:
    # stores a graph
    # use stores mlir.Value to ait.Tensor map
    _value2tensor = None
    _op2parent_block = None
    _im_vals = None

    def __init__(self, func, workdir="./workspace", subgraph_name="model", enable_tf32=False,device="cuda"):
        self.func = func
        self._value2tensor = {}
        self._op2parent_block = {}
        self._im_vals = set()
        self.inputs : list[Tensor] = []
        self.outputs : list[Tensor] = []

        self.tit_module_path = None
        self.tit_model = None

        self.subgraph_name = subgraph_name
        self.ptx_name = subgraph_name + ".ptx"
        self.workdir = workdir
        self.enable_tf32 = enable_tf32
        self.test_name = "./" + subgraph_name
        self.constants = {}
        self.constant_idx = 0

        self.device = device
        
        
        # init arguments
        for idx,i in enumerate(self.func.arguments):
            shaped_type = ir.ShapedType(i.type)
            shape = shaped_type.shape
            dtype = mlir_type_to_torch_str(shaped_type.element_type)
            self._value2tensor[i] = Tensor(shape=shape, dtype=dtype, name=f"input_tensor_{idx}")
            self.inputs.append(self._value2tensor[i])

        # Note: variable `lib_path` is constructed according to the code in compile_model()
        os.makedirs(os.path.join(self.workdir, self.test_name), exist_ok=True)
        ptx_path = os.path.join(self.workdir, self.test_name, self.ptx_name)
        print("TIT module path {} for {}".format(ptx_path, self.ptx_name))
        self.tit_module_path = ptx_path

    def compile(self):
        self._visit_block(self.func.entry_block)
        assert self.tit_module_path is not None
        assert self.tit_kernel is not None

    def _visit_op(self, op):
        # analyze here
        # call bt APIs to create tensor & op
        inputs = list(self._lookup_tensor(i) for i in op.operands)
        if op.operation.name == "func.return":
            self._gen_tit_kernel(inputs)
            return
        if hasattr(op, "operands"):
            outputs = TRITONTemplateIRTranslator.translate(op, inputs)
            if op.operation.name == "mhlo.constant":
                # TODO: need to support FP16
                shaped_type = ir.ShapedType(op.result.type)
                outputs[0]._attrs["name"] = f"const_tensor_{self.constant_idx}"
                np_array = mlir_attr_to_pyobj(op.attributes["value"])
                data = torch.from_numpy(np_array).contiguous().cuda()
                data = data.to(torch_dtype_from_str(mlir_type_to_torch_str(shaped_type.element_type)))
                self.constants[outputs[0]._attrs["name"]] = data
                self.constant_idx += 1
            if op.operation.name != "mhlo.constant":
                for value in op.results:
                    self._im_vals.add(value)
            for output, value in zip(outputs, op.results):
                self._value2tensor[value] = output

        for region in op.operation.regions:
            for block in region.blocks:
                self._visit_block(block)

    def _visit_block(self, block):
        for i in block.operations:
            self._op2parent_block[i] = block
            self._visit_op(i)

    def _lookup_tensor(self, val):
        # return a bt.graph.Tensor
        assert val in self._value2tensor
        return self._value2tensor[val]

    def _gen_tit_kernel(self, results):
        idx = 0
        for out in results:
            out._attrs["name"] = f"output_tensor_{idx}"
            idx += 1
        assert len(results) == 1, "only support single cat op"
        result=results[0]
        self.tit_kernel = compile_kernel(
            op=result,
            device=self.device,
            workdir=self.workdir,
            enable_tf32=self.enable_tf32,
        )
        # kernel rename
        with open(self.tit_module_path, "w") as f:
            f.write(self.tit_kernel.kernel_ptx(self.subgraph_name))
        self.gridsize = self.tit_kernel.gridsize
        self.blocksize = self.tit_kernel.blocksize
        self.smemsize = self.tit_kernel.smemsize

    def _gen_runtime_tensor(self, tensor: Tensor):
        shape = tensor.shape()
        rt_shape = []
        for s in shape:
            if isinstance(s, IntImm):
                rt_shape.append(s.value())
        dtype = torch_dtype_from_str(tensor.dtype())
        if len(rt_shape) == 0:
            return torch.tensor(1).to(torch_dtype_from_str(tensor.dtype())).cuda()
        if dtype == torch.bool:
            return torch.randint(high=1, size=rt_shape, dtype=dtype, device="cuda")
        elif dtype in [torch.int8, torch.int, torch.int16, torch.int32, torch.int64]:
            return torch.randint(high=100, size=rt_shape, dtype=dtype, device="cuda")
        else:
            return torch.randn(*rt_shape, device="cuda").to(torch_dtype_from_str(tensor.dtype()))

    def execute(self, np_inputs, num_trials=1, benchmark=False):
        raise NotImplementedError("TITBuilder.execute() is not implemented yet")

    def benchmark(self, num_trials=5):
        raise NotImplementedError("TITBuilder.benchmark() is not implemented yet")
