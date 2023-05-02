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
from typing import Set

from aitemplate.compiler.base import IntImm, IntVar, Tensor
from aitemplate.compiler import compile_model
from aitemplate.testing import detect_target

from .backend.ait_registry import *
from mhlo_tools.ir_executor.helper import mlir_type_to_dtype

def _torch_dtype_from_str(dtype_name: str) -> torch.dtype:
    _map = {
        "float": torch.float,
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
        # TODO: bfloat16, etc.
    }
    return _map.get(dtype_name, None)

import numpy as np

def _numpy_dtype_to_str(np_dtype: np.dtype) -> str:
    _map = {
        np.single: "float",
        np.half: "float16",
        np.float16: "float16",
        np.float32: "float32",
        np.float64: "float64",
        np.double: "double",
        np.int8: "int8",
        np.int16: "int16",
        np.int32: "int32",
        np.int64: "int64",
        np.bool_: "bool",
    }
    return _map.get(np_dtype, None)

def _init_torch_tensor(AITTensor):
    assert all(map(lambda s: isinstance(s, IntImm), AITTensor.shape()))
    shape_values = [s.value() for s in AITTensor.shape()]
    return torch.empty(shape_values, dtype=_torch_dtype_from_str(AITTensor.dtype()))


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
        #func = self.module.body.operations[0]
        # init arguments
        for i in self.func.arguments:
            shaped_type = ir.ShapedType(i.type)
            shape = shaped_type.shape
            # TODO: dtype and name here?
            dtype = mlir_type_to_dtype(shaped_type.element_type)
            self._value2tensor[i] = Tensor(shape=shape, dtype=_numpy_dtype_to_str(dtype), is_input=True)
            self.inputs.append(self._value2tensor[i])

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
        for key in self._value2tensor:
            ait_tensor = self._value2tensor[key]
            if len(ait_tensor._attrs["src_ops"]) == 0 and not ait_tensor._attrs["is_input"]: # TODO: need a better method here
                torch_tensor = _init_torch_tensor(ait_tensor)
                ait_tensor._attrs["name"] = f"const_tensor_{idx}"
                idx += 1
                constants[ait_tensor._attrs["name"]] = torch_tensor

        for out in results:
            out._attrs["is_output"] = True
        module = compile_model(
            tensor=results,
            target=target,
            workdir=self.workdir,
            test_name=self.test_name,
            dll_name=self.dll_name,
            constants=constants
        )
        print("AIT module path {} for {}".format(module.lib_path, self.dll_name))
        self.ait_module_path = module.lib_path
        self.ait_model = module
        self.outputs = results.copy()

    def _gen_runtime_tensor(self, tensor: Tensor, scale=1.0):
        shape = tensor.shape()
        rt_shape = []
        for s in shape:
            if isinstance(s, IntImm):
                rt_shape.append(s.value())
            elif isinstance(s, IntVar):
                rt_shape.append(random.choice(s._attrs["values"]))
        return torch.randn(*rt_shape).cuda().to(_torch_dtype_from_str(tensor.dtype())) * scale

    def execute(self, np_inputs, num_trials=1, benchmark=False, scale=1.0):
        rt_inputs = {}
        rt_outputs = {}
        for np_input, in_tensor in zip(np_inputs, self.inputs):
            rt_inputs[in_tensor._attrs["name"]] = torch.from_numpy(np_input).contiguous().cuda().to(_torch_dtype_from_str(in_tensor.dtype())) * scale
        for out_tensor in self.outputs:
            rt_outputs[out_tensor._attrs["name"]] = self._gen_runtime_tensor(out_tensor)
        self.ait_model.run_with_tensors(inputs=rt_inputs, outputs=rt_outputs)
        print("finish execution")
        ret = []
        for out_tensor in self.outputs:
            ret.append(rt_outputs[out_tensor._attrs["name"]])
        return ret

    def benchmark(self, num_trials=10):
        rt_inputs = {}
        rt_outputs = {}
        for trial_id in range(num_trials):
            for in_tensor in self.inputs:
                rt_inputs[in_tensor._attrs["name"]] = self._gen_runtime_tensor(in_tensor)
            for out_tensor in self.outputs:
                rt_outputs[out_tensor._attrs["name"]] = self._gen_runtime_tensor(out_tensor)
            self.ait_model.benchmark_with_tensors(inputs=rt_inputs, outputs=rt_outputs)
            print(f"trial {trial_id} finish")
