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

from reporting import TestResult

import brt
from brt.backend import BRTBackend
import byteir
from byteir import ir
from byteir._backend_registry import get_target_device
from byteir.utils import mlir_type_to_np_dtype, np_type_to_torch_type

import torch
import numpy as np
import os
import traceback
import time
from typing import List

np.random.seed(0)
MLIR_TEST_SPECIAL_INPUTS = {
    "cpu@log_plus_one_f16.mlir": [
        np.random.uniform(low=0.5, high=1.0, size=(256, 1)).astype(np.float16)
    ],
    "cpu@convert_f32_i32_special_val.mlir": [
        np.array([[np.inf, -np.inf, np.nan], [1., 999.999, -np.inf]], dtype=np.float32),
    ]
}


class MLIRDataGenerator:
    def __init__(self, mlir_file, target):
        np.random.seed(0)
        self.file_base_name = os.path.basename(mlir_file)
        self.target = target
        context = ir.Context()
        with open(mlir_file, "r") as f:
            self.module = ir.Module.parse(f.read(), context)

    @property
    def entry_func(self):
        return self.module.body.operations[0]

    @property
    def entry_func_name(self) -> str:
        return self.entry_func.name.value    

    def need_special_inputs(self) -> bool:
        key = self.target + "@" + self.file_base_name
        return key in MLIR_TEST_SPECIAL_INPUTS

    def generate_np_inputs(self) -> List[np.ndarray]:
        key = self.target + "@" + self.file_base_name
        if key in MLIR_TEST_SPECIAL_INPUTS:
            return MLIR_TEST_SPECIAL_INPUTS[key]

        entry_func = self.entry_func
        inputs = []
        for type in entry_func.type.inputs:
            shaped_type = ir.ShapedType(type)
            shape = shaped_type.shape
            dtype = mlir_type_to_np_dtype(shaped_type.element_type)
            if dtype == np.bool_:
                inputs.append(np.random.randint(2, size=shape).astype(dtype))
            elif dtype in [
                np.uint8,
                np.int8,
                np.int16,
                np.uint16,
                np.int32,
                np.uint32,
                np.int64,
                np.uint64,
            ]:
                inputs.append(np.random.randint(50, size=shape).astype(dtype))
            else:
                inputs.append(np.random.random(size=shape).astype(dtype))
        return inputs

    def generate_torch_outputs(self, device="cpu") -> List[torch.Tensor]:
        entry_func = self.entry_func
        outputs = []
        for type in entry_func.type.results:
            shaped_type = ir.ShapedType(type)
            shape = shaped_type.shape
            dtype = np_type_to_torch_type(
                mlir_type_to_np_dtype(shaped_type.element_type)
            )
            outputs.append(torch.empty(shape, dtype=dtype, device=device))
        return outputs


def compile_and_run_mlir(mhlo_file, target, workdir, verbose, mode="numerical", unique_name=None, **kwargs):
    if unique_name is None:
        unique_name = os.path.basename(mhlo_file).split(".")[0] + "." + target
    try:
        data_generator = MLIRDataGenerator(mhlo_file, target)
        entry_func_name = data_generator.entry_func_name
        np_inputs = data_generator.generate_np_inputs()

        if mode == "numerical":
            # run golden
            from mhlo_tools.ir_executor import Interpreter

            interp = Interpreter.load_from_file(mhlo_file, is_stablehlo=True)
            golden_outputs = interp.call_function(entry_func_name, np_inputs)

        # byteir compile
        os.makedirs(workdir, exist_ok=True)
        os.makedirs(workdir + f"/{unique_name}", exist_ok=True)
        output_mlir_file_name = f"{workdir}/{unique_name}/{unique_name}.rt.mlir"
        byteir.compile(
            mhlo_file,
            output_mlir_file_name,
            entry_func=entry_func_name,
            target=target,
            verbose=verbose,
        )
    except Exception as e:
        return TestResult(
            unique_name=unique_name,
            compilation_error="".join(
                traceback.format_exception(type(e), e, e.__traceback__)
            ),
            runtime_error=None,
            numerical_error=None,
            performance_result=None,
        )
    # brt runtime
    try:
        cur_device = get_target_device(target)
        brt_backend = BRTBackend(output_mlir_file_name, cur_device)

        torch_inputs = []
        for np_input in np_inputs:
            data = torch.from_numpy(np_input).contiguous().to(cur_device)
            torch_inputs.append(data)
        torch_outputs = data_generator.generate_torch_outputs(cur_device)

        if mode == "numerical":
            brt_backend.run_with_outputs(torch_inputs, torch_outputs)
        else:
            avg_time = brt_backend.profile_with_outputs(torch_inputs, torch_outputs)
            return TestResult(
                unique_name=unique_name,
                compilation_error=None,
                runtime_error=None,
                numerical_error=None,
                performance_result=avg_time,
            )
    except Exception as e:
        return TestResult(
            unique_name=unique_name,
            compilation_error=None,
            runtime_error="".join(
                traceback.format_exception(type(e), e, e.__traceback__)
            ),
            numerical_error=None,
            performance_result=None,
        )
    # numerical check
    try:
        for golden_output, output in zip(golden_outputs, torch_outputs):
            # print("golden output: ", golden_output)
            # print("actual output: ", output.detach().cpu().numpy())
            golden = torch.from_numpy(golden_output).contiguous().to(cur_device)
            torch.testing.assert_close(golden, output)
    except Exception as e:
        return TestResult(
            unique_name=unique_name,
            compilation_error=None,
            runtime_error=None,
            numerical_error="".join(
                traceback.format_exception(type(e), e, e.__traceback__)
            ),
            performance_result=None,
        )
    return TestResult(
        unique_name=unique_name,
        compilation_error=None,
        runtime_error=None,
        numerical_error=None,
        performance_result=None,
    )


def compile_and_run_torch(test, target, workdir, verbose, mode="numerical"):
    from torch_e2e_testing.framework import generate_golden_trace
    import torch_frontend

    unique_name = test.unique_name + "." + target
    cur_device = get_target_device(target)
    # compile
    try:
        golden_trace = generate_golden_trace(test)
        trace_item = golden_trace[0]

        # torch_frontend compile
        torch_inputs = [input.clone().to(cur_device) for input in trace_item.inputs]
        torch_outputs = [
            torch.empty(trace_item.output.shape, dtype=trace_item.output.dtype).to(
                cur_device
            )
        ]
        compiled_graph = torch_frontend.compile(
            test.program_factory(), torch_inputs, "stablehlo"
        )

        # byteir compile
        os.makedirs(workdir, exist_ok=True)
        os.makedirs(workdir + f"/{unique_name}", exist_ok=True)
        mlir_file_name = f"{workdir}/{unique_name}/{unique_name}.stablehlo.mlir"
        output_mlir_file_name = f"{workdir}/{unique_name}/{unique_name}.rt.mlir"
        with open(mlir_file_name, "w+") as fout:
            compiled_graph.operation.print(file=fout, large_elements_limit=None)
        byteir.compile(
            mlir_file_name,
            output_mlir_file_name,
            entry_func="forward",
            target=target,
            verbose=verbose,
        )
    except Exception as e:
        return TestResult(
            unique_name=unique_name,
            compilation_error="".join(
                traceback.format_exception(type(e), e, e.__traceback__)
            ),
            runtime_error=None,
            numerical_error=None,
            performance_result=None,
        )

    # runtime
    try:
        brt_backend = BRTBackend(output_mlir_file_name, cur_device)
        if mode == "numerical":
            brt_backend.run_with_outputs(torch_inputs, torch_outputs)
        else:
            avg_time = brt_backend.profile_with_outputs(torch_inputs, torch_outputs)
            return TestResult(
                unique_name=unique_name,
                compilation_error=None,
                runtime_error=None,
                numerical_error=None,
                performance_result=avg_time,
            )
    except Exception as e:
        return TestResult(
            unique_name=unique_name,
            compilation_error=None,
            runtime_error="".join(
                traceback.format_exception(type(e), e, e.__traceback__)
            ),
            numerical_error=None,
            performance_result=None,
        )
    # numerical check
    try:
        golden_output = trace_item.output.detach().cpu()
        actual_output = torch_outputs[0].detach().cpu()
        torch.testing.assert_close(golden_output, actual_output)
    except Exception as e:
        return TestResult(
            unique_name=unique_name,
            compilation_error=None,
            runtime_error=None,
            numerical_error="".join(
                traceback.format_exception(type(e), e, e.__traceback__)
            ),
            performance_result=None,
        )
    return TestResult(
        unique_name=unique_name,
        compilation_error=None,
        runtime_error=None,
        numerical_error=None,
        performance_result=None,
    )
