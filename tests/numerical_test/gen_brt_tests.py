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

#!/usr/bin/python3
import argparse
import os
import re
import sys
import shutil
import traceback

import numpy as np
from execute import MLIRDataGenerator
from reporting import TestResult, report_results
from testset import CPU_MLIR_TEST_DIR, CPU_MLIR_TEST_SET, CPU_XFAIL_SET
import byteir

parser = argparse.ArgumentParser()
parser.add_argument("--target",
                    type=str,
                    default="cpu",
                    choices=["all", "cuda", "cpu"],
                    help="target device name")
parser.add_argument("--output_dir",
                    type=str,
                    default="./local_golden",
                    help="output directory path")
parser.add_argument("-f",
                    "--filter",
                    type=str,
                    default=".*",
                    help="Regular expression specifying which tests to include in this run.")
parser.add_argument("--byre_serial_version",
                    type=str,
                    default="1.0.0",
                    choices=["1.0.0"],
                    help="Byre serialization target version")
args = parser.parse_args()

# Unsupported ops
EXCLUDE_MLIR_CPU_TESTS = [
    "custom_call_tf_UpperBound.mlir",
    "rng.mlir",
]

def gen_golden_mlir(mhlo_file, target, golden_dir, num=2):
    """
    Arguements:
        @param mhlo_file: Source stablehlo/mhlo file.
        @param target: Target name like `cpu`,`cuda`
        @param num: Numbers of generated golden in/output, default to 5.
    """

    def save_np_data(fpath: str, data):
        np.savez(fpath, *data)

    file_base_name = os.path.basename(mhlo_file).split(".")[0]
    unique_name = file_base_name + "." + target
    try:
        data_generator = MLIRDataGenerator(mhlo_file, target)
        func_name = data_generator.entry_func_name

        WORK_FOLDER = golden_dir + f"/{unique_name}"
        os.makedirs(WORK_FOLDER, exist_ok=True)

        # if need special inputs, only iterate 1 time
        if data_generator.need_special_inputs():
            num = 1

        for idx in range(0, num):
            np_inputs = data_generator.generate_np_inputs()

            # run golden
            from mhlo_tools.ir_executor import Interpreter
            interp = Interpreter.load_from_file(mhlo_file, is_stablehlo=True)
            golden_outputs = interp.call_function(func_name, np_inputs)

            # dump to local file
            save_np_data(WORK_FOLDER + f"/inputs.{str(idx)}.npz", np_inputs)
            save_np_data(WORK_FOLDER + f"/outputs.{str(idx)}.npz", golden_outputs)

            del np_inputs, golden_outputs

        # byteir compile
        output_mlir_file_name = f"{WORK_FOLDER}/{unique_name}.rt.mlirbc"
        byteir.compile(
            mhlo_file, output_mlir_file_name, entry_func=func_name, target=target
        )
        # cp orininal mlir file
        shutil.copy(
            mhlo_file,
            f"{WORK_FOLDER}/{file_base_name}.stablehlo.mlir",
        )
        # serialize to stablehlo bytecode
        from byteir._mlir_libs._stablehlo import serialize_portable_artifact, get_current_version
        bytes = serialize_portable_artifact(data_generator.module.operation.get_asm(), get_current_version())
        with open(f"{WORK_FOLDER}/{file_base_name}.stablehlo.mlirbc", "wb") as f:
            f.write(bytes)

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

    res = TestResult(
        unique_name=unique_name,
        compilation_error=None,
        runtime_error=None,
        numerical_error=None,
        performance_result=None,
    )

    return res



def gen_mlir_cpu_golden():
    directory = CPU_MLIR_TEST_DIR
    cpu_target = "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    golden_dir = f"{args.output_dir}/CPU_BYRE_{args.byre_serial_version.replace('.', '_')}"
    os.makedirs(golden_dir, exist_ok=True)

    mlir_tests = []
    for filename in os.listdir(directory):
        if filename.startswith('.'):
            continue
        if not re.match(args.filter, filename):
            continue
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f) and filename in (CPU_MLIR_TEST_SET - CPU_XFAIL_SET):
            mlir_tests.append(f)

    results = []
    for test in mlir_tests:
        fpath = test
        res = gen_golden_mlir(fpath,
                              cpu_target,
                              golden_dir)
        results.append(res)
    return results


def gen():
    results = []
    if args.target == 'all':
        results = gen_mlir_cpu_golden()
    elif args.target == "cpu":
        results = gen_mlir_cpu_golden()
    elif args.target == "cuda":
        pass
    failed = report_results(results)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    gen()
