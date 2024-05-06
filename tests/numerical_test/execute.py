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

from torch_e2e_testing.framework import generate_golden_trace
import brt
import byteir
from byteir.registry import get_target_device
from mhlo_tools.ir_executor import Interpreter
from mhlo_tools.mlir import ir
import torch
from torch.cuda.memory import caching_allocator_alloc, caching_allocator_delete
import numpy as np
from mhlo_tools.ir_executor.helper import (
    mlir_type_to_dtype
)
import os
import torch_frontend
import traceback
from utils import TestResult


def np_type_to_torch_type(np_dtype):
    _map = {
        np.single: torch.float32,
        np.half: torch.float16,
        np.float16: torch.float16,
        np.float32: torch.float32,
        np.float64: torch.float64,
        np.double: torch.float64,
        np.int8: torch.int8,
        np.int16: torch.int16,
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.bool_: torch.bool,
    }
    return _map.get(np_dtype, None)


def generate_np_inputs(interp):
    module = interp._mod
    entry_func = module.body.operations[0]
    ret = []
    for arg in entry_func.arguments:
        shaped_type = ir.ShapedType(arg.type)
        shape = shaped_type.shape
        dtype = mlir_type_to_dtype(shaped_type.element_type)
        if dtype == np.bool_:
            ret.append(np.random.randint(2, size=shape).astype(dtype))
        elif dtype in [np.uint8, np.int8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64]:
            ret.append(np.random.randint(50, size=shape).astype(dtype))
        else:
            ret.append(np.random.random(size=shape).astype(dtype))
    return ret


def generate_torch_outputs(interp, device: str = "cuda"):
    module = interp._mod
    entry_func = module.body.operations[0]
    torch_outputs = []
    for op in entry_func.entry_block.operations:
        if op.operation.name == "func.return":
            for val in op.operation.operands:
                shaped_type = ir.ShapedType(val.type)
                shape = shaped_type.shape
                dtype = np_type_to_torch_type(
                    mlir_type_to_dtype(shaped_type.element_type))
                torch_outputs.append(torch.empty(
                    shape, dtype=dtype, device=device))
    return torch_outputs


def get_entry_func_name(interp):
    module = interp._mod
    entry_func = module.body.operations[0].name.value
    return entry_func


def compile_and_run_mlir(mhlo_file, target):
    np.random.seed(0)
    try:
        interp = Interpreter.load_from_file(mhlo_file)
        np_inputs = generate_np_inputs(interp)
        func_name = get_entry_func_name(interp)
        unique_name = os.path.basename(mhlo_file).split('.')[0]
        unique_name = unique_name + "." + target

        # run golden
        golden_outputs = interp.call_function(func_name, np_inputs)

        # byteir compile
        TEMP_FOLDER = "./local_test"
        os.makedirs(TEMP_FOLDER, exist_ok=True)
        os.makedirs(TEMP_FOLDER + f"/{unique_name}", exist_ok=True)
        output_mlir_file_name = f'{TEMP_FOLDER}/{unique_name}/{unique_name}.rt.mlir'
        byteir.compile(mhlo_file, output_mlir_file_name,
                       entry_func=func_name, target=target)
    except Exception as e:
        return TestResult(unique_name=mhlo_file,
                          compilation_error="".join(
                              traceback.format_exception(
                                  type(e), e, e.__traceback__)),
                          runtime_error=None,
                          numerical_error=None)
    # brt runtime
    try:
        cur_device = get_target_device(target)
        _allocator_alloc = None
        _allocator_delete = None
        _stream = None
        if "cuda" == cur_device:
            _allocator_alloc = caching_allocator_alloc
            _allocator_delete = caching_allocator_delete
            _stream = torch.cuda.current_stream()._as_parameter_.value

        session = brt.Session(device=cur_device.upper(),
                              alloc_func=_allocator_alloc,
                              free_func=_allocator_delete)
        session.load(output_mlir_file_name)
        req = session.new_request_context(_stream)
        torch_inputs = []
        torch_outputs = []
        for np_input in np_inputs:
            data = torch.from_numpy(np_input).contiguous().to(cur_device)
            data = data.to(np_type_to_torch_type(np_input.dtype))
            torch_inputs.append(data)

        torch_outputs = generate_torch_outputs(interp, cur_device)
        for offset, arg in zip(session.get_input_arg_offsets(), torch_inputs):
            assert list(session.get_static_shape(offset)) == list(arg.shape)
        for offset, ret in zip(session.get_output_arg_offsets(), torch_outputs):
            assert list(session.get_static_shape(offset)) == list(ret.shape)

        for i, tensor in zip(session.get_input_arg_offsets(), torch_inputs):
            req.bind_arg(i, tensor.data_ptr())
        for i, tensor in zip(session.get_output_arg_offsets(), torch_outputs):
            req.bind_arg(i, tensor.data_ptr())

        req.finish_io_binding()
        req.run()
        req.sync()
    except Exception as e:
        return TestResult(unique_name=mhlo_file,
                          compilation_error=None,
                          runtime_error="".join(
                              traceback.format_exception(
                                  type(e), e, e.__traceback__)),
                          numerical_error=None)
    # compare outputs
    try:
        for golden_output, output in zip(golden_outputs, torch_outputs):
            # print("golden output: ", golden_output)
            # print("actual output: ", output.detach().cpu().numpy())
            data = torch.from_numpy(golden_output).contiguous().to(cur_device)
            data = data.to(np_type_to_torch_type(golden_output.dtype))
            torch.testing.assert_close(data, output)
        # assert(np.allclose(golden_output, output.detach().cpu().numpy()))
    except Exception as e:
        return TestResult(unique_name=mhlo_file,
                          compilation_error=None,
                          runtime_error=None,
                          numerical_error="".join(
                              traceback.format_exception(
                                  type(e), e, e.__traceback__)))
    return TestResult(unique_name=mhlo_file,
                      compilation_error=None,
                      runtime_error=None,
                      numerical_error=None)


def compile_and_run_torch(test, target):
    # compile
    try:
        golden_trace = generate_golden_trace(test)
        trace_item = golden_trace[0]

        torch_inputs = [input.clone().cuda() for input in trace_item.inputs]
        torch_outputs = [torch.empty_like(trace_item.output).cuda()]
        compiled_graph = torch_frontend.compile(
            test.program_factory(), torch_inputs, 'stablehlo')

        TEMP_FOLDER = "./local_test"
        os.makedirs(TEMP_FOLDER, exist_ok=True)
        os.makedirs(TEMP_FOLDER + f"/{test.unique_name}", exist_ok=True)
        mlir_file_name = f'{TEMP_FOLDER}/{test.unique_name}/{test.unique_name}.mhlo.mlir'
        output_mlir_file_name = f'{TEMP_FOLDER}/{test.unique_name}/{test.unique_name}.rt.mlir'
        with open(mlir_file_name, "w+") as fout:
            compiled_graph.operation.print(file=fout,
                                           large_elements_limit=None)
        byteir.compile(mlir_file_name, output_mlir_file_name,
                       entry_func="forward", target=target)
    except Exception as e:
        return TestResult(unique_name=test.unique_name,
                          compilation_error="".join(
                              traceback.format_exception(
                                  type(e), e, e.__traceback__)),
                          runtime_error=None,
                          numerical_error=None)

    # runtime
    try:
        session = brt.Session(alloc_func=caching_allocator_alloc,
                               free_func=caching_allocator_delete)
        session.load(output_mlir_file_name)
        req = session.new_request_context(
            torch.cuda.current_stream()._as_parameter_.value)

        for offset, arg in zip(session.get_input_arg_offsets(), torch_inputs):
            assert list(session.get_static_shape(offset)) == list(arg.shape)
        for offset, ret in zip(session.get_output_arg_offsets(), torch_outputs):
            assert list(session.get_static_shape(offset)) == list(ret.shape)

        for i, tensor in zip(session.get_input_arg_offsets(), torch_inputs):
            req.bind_arg(i, tensor.data_ptr())
        for i, tensor in zip(session.get_output_arg_offsets(), torch_outputs):
            req.bind_arg(i, tensor.data_ptr())

        req.finish_io_binding()
        req.run()
        req.sync()
    except Exception as e:
        return TestResult(unique_name=test.unique_name,
                          compilation_error=None,
                          runtime_error="".join(
                              traceback.format_exception(
                                  type(e), e, e.__traceback__)),
                          numerical_error=None)

    # numerical check
    golden_output = trace_item.output.detach().cpu()
    actual_output = torch_outputs[0].detach().cpu()

    try:
        torch.testing.assert_close(golden_output, actual_output)
    except Exception as e:
        return TestResult(unique_name=test.unique_name,
                          compilation_error=None,
                          runtime_error=None,
                          numerical_error="".join(
                              traceback.format_exception(
                                  type(e), e, e.__traceback__)))
    return TestResult(unique_name=test.unique_name,
                      compilation_error=None,
                      runtime_error=None,
                      numerical_error=None)
