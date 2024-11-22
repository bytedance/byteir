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

from execute import compile_and_run_torch, compile_and_run_mlir
from reporting import report_results
from torch_e2e_testing.registry import (
    GLOBAL_TORCH_TEST_REGISTRY,
    GLOBAL_TORCH_TEST_REGISTRY_NAMES,
)
from testset import CPU_MLIR_TEST_DIR, CUDA_MLIR_TEST_DIR
from testset import CPU_ALL_SET, CUDA_ALL_SET, CUDA_AIT_ALL_SET, CUDA_AIT_SM80PLUS_SET

##### TEST SET CONFIG #######
TEST_SET = {
    "cpu": CPU_ALL_SET,
    "cuda": CUDA_ALL_SET,
    "cuda_with_ait": CUDA_AIT_ALL_SET,
}

def get_local_gpu_arch():
    from byteir.utils import detect_gpu_arch_with_nvidia_smi
    gpu_arch = detect_gpu_arch_with_nvidia_smi()
    assert gpu_arch != None
    assert gpu_arch.startswith("sm_")
    gpu_arch = int(gpu_arch[3:])
    return gpu_arch

def run(target, filter, workdir, mode="numerical", verbose=False):
    if target == "dynamo":
        from torch_dynamo_e2e_testing.execute import run_torch_dynamo_tests
        gpu_arch = get_local_gpu_arch()
        # TODO(zzk): use test infra for dynamo tests
        run_torch_dynamo_tests(gpu_arch)
        return []

    test_set = TEST_SET[target]
    if target != "cpu":
        gpu_arch = get_local_gpu_arch()
        if target == "cuda_with_ait" and gpu_arch < 80:
            test_set -= CUDA_AIT_SM80PLUS_SET

    results = []
    for test in test_set:
        if not re.match(filter, test):
            continue
        if test in GLOBAL_TORCH_TEST_REGISTRY_NAMES:
            results.append(
                compile_and_run_torch(
                    GLOBAL_TORCH_TEST_REGISTRY[test], target, workdir, verbose, mode
                )
            )
        else:
            if target == "cpu":
                results.append(
                    compile_and_run_mlir(
                        os.path.join(CPU_MLIR_TEST_DIR, test), target, workdir, verbose, mode
                    )
                )
            else:
                results.append(
                    compile_and_run_mlir(
                        os.path.join(CUDA_MLIR_TEST_DIR, test), target, workdir, verbose, mode
                    )
                )
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=str,
        default="cuda",
        choices=[
            "all",
            "cpu",
            "cuda",
            "cuda_with_ait",
            "dynamo",
            "native_torch",
        ],
        help="target backend to run",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="numerical",
        choices=["numerical", "profile"],
        help="testing mode, `numerical` means numerical test, `profile` means performance test",
    )
    parser.add_argument(
        "-f",
        "--filter",
        type=str,
        default=".*",
        help="Regular expression specifying which tests to include in this run.",
    )
    parser.add_argument(
        "-s",
        "--sequential",
        default=False,
        action="store_true",
        help="Run tests sequentially rather than in parallel",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="Report test results with additional detail",
    )
    parser.add_argument(
        "--workdir",
        type=str,
        default="./local_test",
        help="Work directory to save compiled outputs",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    results = []
    if args.target == "all":
        for target in ["cpu", "cuda", "cuda_with_ait"]:#, "dynamo"]:
            results += run(target, args.filter, args.workdir)
    else:
        results += run(args.target, args.filter, args.workdir, mode=args.mode, verbose=args.verbose)

    failed = report_results(results)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
