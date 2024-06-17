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
from reporting import TestResult

import torch_frontend
import brt
import byteir
from byteir._backend_registry import get_target_device
from byteir.utils import mlir_type_to_np_dtype, np_type_to_torch_type

from mhlo_tools.ir_executor import Interpreter
from mhlo_tools.mlir import ir

import torch
import numpy as np
import os
import shutil
import traceback

def generate_np_inputs(interp, mode: str = "", low = 0.0, high = 1.0):
    module = interp._mod
    entry_func = module.body.operations[0]
    ret = []
    for arg in entry_func.arguments:
        shaped_type = ir.ShapedType(arg.type)
        shape = shaped_type.shape
        dtype = mlir_type_to_np_dtype(shaped_type.element_type)
        if dtype == np.bool_:
            ret.append(np.random.randint(2, size=shape).astype(dtype))
        elif dtype in [np.uint8, np.int8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64]:
            ret.append(np.random.randint(50, size=shape).astype(dtype))
        else:
            if mode == "uniform":
                ret.append(np.random.uniform(low=low, high=high, size=shape).astype(dtype))
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
                    mlir_type_to_np_dtype(shaped_type.element_type))
                torch_outputs.append(torch.empty(
                    shape, dtype=dtype, device=device))
    return torch_outputs


def get_entry_func_name(interp):
    module = interp._mod
    entry_func = module.body.operations[0].name.value
    return entry_func

def gen_golden_mlir(mhlo_file, target, **kwargs):
    """
    Arguements:
        @param mhlo_file: Source stablehlo/mhlo file.
        @param target: Target name like `cpu`,`cuda`
        @param num: Numbers of generated golden in/output, default to 5.
        @param mode:  The data distribution of inputs.
        @param low/hing: The range of generated inputs data.
    """
    def save_np_data(fpath: str, data):
        np.save(fpath, data)

    np.random.seed(0)
    try:
        if target.lower() == "cpu":
            interp = Interpreter.load_from_file(mhlo_file, is_stablehlo=True)
        else:
            interp = Interpreter.load_from_file(mhlo_file)
        func_name = get_entry_func_name(interp)
        unique_name = os.path.basename(mhlo_file).split('.')[0]
        unique_name = unique_name + "." + target
        iter_number = kwargs["num"] if "num" in kwargs else 5

        WORK_FOLDER = kwargs["golden_dir"] if "golden_dir" in kwargs else "./local_test"
        WORK_FOLDER = WORK_FOLDER + f"/{unique_name}"
        os.makedirs(WORK_FOLDER, exist_ok=True)

        for idx in range(0, iter_number):
            if "mode" in kwargs:
                input_mode = kwargs["mode"]
                low = kwargs["low"] if "low" in kwargs else None
                high = kwargs["high"] if "high" in kwargs else None
                np_inputs = generate_np_inputs(interp, input_mode, low, high)
            else:
                np_inputs = generate_np_inputs(interp)

            # run golden
            golden_outputs = interp.call_function(func_name, np_inputs)

            # dump to local file
            save_np_data(WORK_FOLDER + f"/input_{str(idx)}.npy", np_inputs)
            save_np_data(WORK_FOLDER +  f"/output_{str(idx)}.npy", golden_outputs)

            del np_inputs, golden_outputs

        # byteir compile
        output_mlir_file_name = f'{WORK_FOLDER}/{unique_name}.rt.mlir'
        byteir.compile(mhlo_file, output_mlir_file_name,
                       entry_func=func_name, target=target)

        # cp orininal mlir file
        shutil.copy(mhlo_file, f"{WORK_FOLDER}/{os.path.basename(mhlo_file).split('.')[0]}.stablehlo.mlir")

        
    except Exception as e:
        return TestResult(unique_name=mhlo_file,
                          compilation_error="".join(
                              traceback.format_exception(
                                  type(e), e, e.__traceback__)),
                          runtime_error=None,
                          numerical_error=None,
                          performance_result=None)

    res = TestResult(unique_name=mhlo_file,
                     compilation_error=None,
                     runtime_error=None,
                     numerical_error=None,
                     performance_result=None)

    return res


class BRTBackend:
    def __init__(self, device, brt_file_path):
        from torch.cuda.memory import caching_allocator_alloc, caching_allocator_delete
        _allocator_alloc = caching_allocator_alloc if device == "cuda" else None
        _allocator_delete = caching_allocator_delete if device == "cuda" else None
        _stream = torch.cuda.current_stream()._as_parameter_.value if device == "cuda" else None
        self.session = brt.Session(device=device.upper(),
                                   alloc_func=_allocator_alloc,
                                   free_func=_allocator_delete)
        self.session.load(brt_file_path)
        self.req = self.session.new_request_context(_stream)

    def execute(self, inputs, outputs):
        for offset, arg in zip(self.session.get_input_arg_offsets(), inputs):
            assert list(self.session.get_static_shape(offset)) == list(arg.shape)
            self.req.bind_arg(offset, arg.data_ptr())
        for offset, ret in zip(self.session.get_output_arg_offsets(), outputs):
            assert list(self.session.get_static_shape(offset)) == list(ret.shape)
            self.req.bind_arg(offset, ret.data_ptr())
        self.req.finish_io_binding()
        self.req.run()
        self.req.sync()

    def profile(self, inputs, outputs, warmup_trials=10, run_trials=50):
        for offset, arg in zip(self.session.get_input_arg_offsets(), inputs):
            assert list(self.session.get_static_shape(offset)) == list(arg.shape)
            self.req.bind_arg(offset, arg.data_ptr())
        for offset, ret in zip(self.session.get_output_arg_offsets(), outputs):
            assert list(self.session.get_static_shape(offset)) == list(ret.shape)
            self.req.bind_arg(offset, ret.data_ptr())
        self.req.finish_io_binding()
        
        for _ in range(warmup_trials):
            self.req.run()
        self.req.sync()

        import time
        start = time.time()
        for _ in range(run_trials):
            self.req.run()
            self.req.sync()
        end = time.time()
        return ((end - start) * 1000) / run_trials


def compile_and_run_mlir(mhlo_file, target, verbose, mode="numerical", **kwargs):
    np.random.seed(0)
    try:
        if target.lower() == "cpu":
            interp = Interpreter.load_from_file(mhlo_file, is_stablehlo=True)
        else:
            interp = Interpreter.load_from_file(mhlo_file)
        if "random_mode" in kwargs:
            input_mode = kwargs["random_mode"]
            low = kwargs["low"] if "low" in kwargs else None
            high = kwargs["high"] if "high" in kwargs else None
            np_inputs = generate_np_inputs(interp, input_mode, low, high)
        else:
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
                       entry_func=func_name, target=target, verbose=verbose)
    except Exception as e:
        return TestResult(unique_name=mhlo_file,
                          compilation_error="".join(
                              traceback.format_exception(
                                  type(e), e, e.__traceback__)),
                          runtime_error=None,
                          numerical_error=None,
                          performance_result=None)
    # brt runtime
    try:
        cur_device = get_target_device(target)
        brt_backend = BRTBackend(cur_device, output_mlir_file_name)

        torch_inputs = []
        torch_outputs = []
        for np_input in np_inputs:
            data = torch.from_numpy(np_input).contiguous().to(cur_device)
            data = data.to(np_type_to_torch_type(np_input.dtype))
            torch_inputs.append(data)
        torch_outputs = generate_torch_outputs(interp, cur_device)

        if mode == "numerical":
            brt_backend.execute(torch_inputs, torch_outputs)
        else:
            avg_time = brt_backend.profile(torch_inputs, torch_outputs)
    except Exception as e:
        return TestResult(unique_name=mhlo_file,
                          compilation_error=None,
                          runtime_error="".join(
                              traceback.format_exception(
                                  type(e), e, e.__traceback__)),
                          numerical_error=None,
                          performance_result=None)
    if mode == "profile":
        return TestResult(unique_name=mhlo_file,
                          compilation_error=None,
                          runtime_error=None,
                          numerical_error=None,
                          performance_result=avg_time)
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
                                  type(e), e, e.__traceback__)),
                          performance_result=None)
    return TestResult(unique_name=mhlo_file,
                      compilation_error=None,
                      runtime_error=None,
                      numerical_error=None,
                      performance_result=None)


def compile_and_run_torch(test, target, verbose, mode="numerical"):
    # compile
    try:
        golden_trace = generate_golden_trace(test)
        trace_item = golden_trace[0]

        torch_inputs = [input.clone().cuda() for input in trace_item.inputs]
        torch_outputs = [torch.empty(trace_item.output.shape, dtype=trace_item.output.dtype).cuda()]
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
                       entry_func="forward", target=target, verbose=verbose)
    except Exception as e:
        return TestResult(unique_name=test.unique_name,
                          compilation_error="".join(
                              traceback.format_exception(
                                  type(e), e, e.__traceback__)),
                          runtime_error=None,
                          numerical_error=None,
                          performance_result=None)

    # runtime
    try:
        brt_backend = BRTBackend("cuda", output_mlir_file_name)
        if mode == "numerical":
            brt_backend.execute(torch_inputs, torch_outputs)
        else:
            avg_time = brt_backend.profile(torch_inputs, torch_outputs)
    except Exception as e:
        return TestResult(unique_name=test.unique_name,
                          compilation_error=None,
                          runtime_error="".join(
                              traceback.format_exception(
                                  type(e), e, e.__traceback__)),
                          numerical_error=None,
                          performance_result=None)

    if mode == "profile":
        return TestResult(unique_name=test.unique_name,
                          compilation_error=None,
                          runtime_error=None,
                          numerical_error=None,
                          performance_result=avg_time)

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
                                  type(e), e, e.__traceback__)),
                          performance_result=None)
    return TestResult(unique_name=test.unique_name,
                      compilation_error=None,
                      runtime_error=None,
                      numerical_error=None,
                      performance_result=None)
