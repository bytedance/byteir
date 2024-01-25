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
from execute import compile_and_run_mlir, compile_and_run_torch
from torch_e2e_testing.registry import GLOBAL_TORCH_TEST_REGISTRY
from torch_e2e_testing.test_suite import register_all_torch_tests
from torch_dynamo_e2e_testing.execute import run_torch_dynamo_tests
from utils import report_results
import sys
from subprocess import PIPE, Popen

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filter", default=".*", help="""
Regular expression specifying which tests to include in this run.
""")
parser.add_argument("--target", type=str, default="cuda_with_ait",
                    choices=["ait", "cuda", "cuda_with_ait_aggressive"], help="target device name")
parser.add_argument("-c", "--config", default="all",
                    choices=["all", "mlir", "torch", "dynamo"], help="test sets to run.")
args = parser.parse_args()

EXCLUDE_MLIR_TESTS = []

EXCLUDE_TORCH_TESTS = []

SM80_PLUS_TESTS = [
    "dot_f32.mlir",
    "bmm_rrr_permute_f16.mlir",
    "bmm_rrr_permute_f32.mlir",
    "MatmulF32Module_basic",
    "BatchMatmulAddF32Module_basic",
    "BatchMatmulF32Module_basic",
]


def _detect_cuda_with_nvidia_smi():
    try:
        proc = Popen(
            ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv"],
            stdout=PIPE,
            stderr=PIPE,
        )
        stdout, stderr = proc.communicate()
        stdout = stdout.decode("utf-8")
        sm_names = {
            70: ["V100"],
            75: ["T4", "Quadro T2000"],
            80: ["PG509", "A100", "A10", "RTX 30", "A30", "RTX 40", "A16"],
            90: ["H100"],
        }
        for sm, names in sm_names.items():
            if any(name in stdout for name in names):
                return sm
        return None
    except Exception:
        return None


def is_test_supported(arch, test_name):
    # TODO: other arch
    if arch < 80:
        return test_name not in SM80_PLUS_TESTS
    return True


def run_mlir_test(arch):
    directory = os.path.dirname(os.path.realpath(__file__))
    directory = directory + "/mlir_tests/ops"
    mlir_tests = []

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f) and re.match(args.filter, filename):
            if filename not in EXCLUDE_MLIR_TESTS and is_test_supported(arch, filename):
                mlir_tests.append(f)

    results = []
    for test in mlir_tests:
        results.append(compile_and_run_mlir(test, args.target))
    return results


def run_torch_test(arch):
    tests = [
        test for test in GLOBAL_TORCH_TEST_REGISTRY
        if re.match(args.filter, test.unique_name)
        and test.unique_name not in EXCLUDE_TORCH_TESTS
        and is_test_supported(arch, test.unique_name)
    ]
    results = []
    for test in tests:
        results.append(compile_and_run_torch(test, args.target))
    return results


def main():
    results = []
    arch = _detect_cuda_with_nvidia_smi()
    assert (arch != None)
    if args.config == 'all':
        results = run_mlir_test(arch)
        results = results + run_torch_test(arch)
        run_torch_dynamo_tests(arch)
    elif args.config == 'mlir':
        results = run_mlir_test(arch)
    elif args.config == 'torch':
        results = run_torch_test(arch)
    elif args.config == 'dynamo':
        # TODO(zzk): use test infra for dynamo tests
        run_torch_dynamo_tests(arch)
        pass
    failed = report_results(results)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    register_all_torch_tests()
    main()
