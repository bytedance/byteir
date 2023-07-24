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
import numpy as np
from typing import Set

from aitemplate.compiler.base import IntImm, IntVar, Tensor, _NumpyConstantTensorData
from aitemplate.compiler import compile_model
from aitemplate.testing import detect_target

from .backend.ait_registry import *

from byteir.utils import mlir_type_to_torch_str, torch_dtype_from_str

def _init_torch_tensor(AITTensor):
    assert all(map(lambda s: isinstance(s, IntImm), AITTensor.shape()))
    shape_values = [s.value() for s in AITTensor.shape()]
    return torch.empty(shape_values, dtype=torch_dtype_from_str(AITTensor.dtype()))


# TODO: merge common part of ait & bt builder into a base class
class ait_builder:
    # stores a graph
    # use stores mlir.Value to ait.Tensor map
    _value2tensor = None
    _op2parent_block = None
    _im_vals = None

    def __init__(self, func, workdir="./workspace", subgraph_name="model"):
        self.func = func
        self._value2tensor = {}
        self._op2parent_block = {}
        self._im_vals = set()
        self.inputs : list[Tensor] = []
        self.outputs : list[Tensor] = []

        self.ait_module_path = None
        self.ait_model = None

        self.workdir = workdir
        self.test_name = "./" + subgraph_name
        self.dll_name = subgraph_name + ".so"
        self.constants = {}
        self.constant_idx = 0
        #func = self.module.body.operations[0]
        # init arguments
        idx = 0
        for i in self.func.arguments:
            shaped_type = ir.ShapedType(i.type)
            shape = shaped_type.shape
            dtype = mlir_type_to_torch_str(shaped_type.element_type)
            self._value2tensor[i] = Tensor(shape=shape, dtype=dtype, is_input=True, name=f"input_tensor_{idx}")
            self.inputs.append(self._value2tensor[i])
            idx += 1

        self._visit_block(self.func.entry_block)
        assert self.ait_module_path is not None
        assert self.ait_model is not None

    def _visit_op(self, op):
        # analyze here
        # call bt APIs to create tensor & op
        inputs = list(self._lookup_tensor(i) for i in op.operands)
        if op.operation.name == "func.return":
            self._gen_ait_module(inputs)
            return
        if hasattr(op, "operands"):
            outputs = AITemplateIRTranslator.translate(op, inputs)
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

    def _gen_ait_module(self, results):
        target = detect_target()
        constants = {}

        idx = 0
        for out in results:
            out._attrs["is_output"] = True
            out._attrs["name"] = f"output_tensor_{idx}"
            idx += 1

        module = compile_model(
            tensor=results,
            target=target,
            workdir=self.workdir,
            test_name=self.test_name,
            dll_name=self.dll_name,
            constants=self.constants
        )
        print("AIT module path {} for {}".format(module.lib_path, self.dll_name))
        self.ait_module_path = module.lib_path
        self.ait_model = module
        self.outputs = results.copy()

    def _gen_runtime_tensor(self, tensor: Tensor):
        shape = tensor.shape()
        rt_shape = []
        for s in shape:
            if isinstance(s, IntImm):
                rt_shape.append(s.value())
            elif isinstance(s, IntVar):
                rt_shape.append(random.choice(s._attrs["values"]))
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
        rt_inputs = {}
        rt_outputs = {}
        for np_input, in_tensor in zip(np_inputs, self.inputs):
            rt_inputs[in_tensor._attrs["name"]] = torch.from_numpy(np_input).contiguous().cuda().to(torch_dtype_from_str(in_tensor.dtype()))
        for out_tensor in self.outputs:
            rt_outputs[out_tensor._attrs["name"]] = self._gen_runtime_tensor(out_tensor)
        self.ait_model.run_with_tensors(inputs=rt_inputs, outputs=rt_outputs)
        print("finish execution")
        ret = []
        for out_tensor in self.outputs:
            ret.append(rt_outputs[out_tensor._attrs["name"]])
        return ret

    def benchmark(self, num_trials=5):
        rt_inputs = {}
        rt_outputs = {}
        for trial_id in range(num_trials):
            for in_tensor in self.inputs:
                rt_inputs[in_tensor._attrs["name"]] = self._gen_runtime_tensor(in_tensor)
            for out_tensor in self.outputs:
                rt_outputs[out_tensor._attrs["name"]] = self._gen_runtime_tensor(out_tensor)
            self.ait_model.benchmark_with_tensors(inputs=rt_inputs, outputs=rt_outputs)
            print(f"trial {trial_id} finish")
