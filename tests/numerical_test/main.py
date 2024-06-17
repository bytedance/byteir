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
from torch_e2e_testing.registry import GLOBAL_TORCH_TEST_REGISTRY
from torch_e2e_testing.test_suite import register_all_torch_tests
from reporting import report_results
import sys

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

EXCLUDE_MLIR_TESTS = []

# Unsupported ops
EXCLUDE_MLIR_CPU_TESTS = [
    "custom_call_tf_UpperBound.mlir",
    "rng.mlir",
]

EXCLUDE_TORCH_TESTS = []

MLIR_CPU_SPECIAL_INPUTS = {
    "log_plus_one.mlir": ["uniform", 0.5, 1.0],
}

SM80_PLUS_TESTS = [
    "dot_f32.mlir",
    "bmm_rrr_permute_f16.mlir",
    "bmm_rrr_permute_f32.mlir",
    "MatmulF32Module_basic",
    "BatchMatmulAddF32Module_basic",
    "BatchMatmulF32Module_basic",
]


def _get_test_files_from_dir(directory):
    test_files = []
    for filename in os.listdir(directory):
        if filename.startswith("."):
            continue
        if os.path.isfile(os.path.join(directory, filename)):
            test_files.append(filename)
    return test_files


def _is_gpu_test_supported(gpu_arch, test_name):
    # TODO: other arch
    if gpu_arch < 80:
        return test_name not in SM80_PLUS_TESTS
    return True


def run_mlir_test(target, gpu_arch, filter, verbose, mode):
    from execute import compile_and_run_mlir
    directory = os.path.join(CUR_DIR, "mlir_tests", "ops")
    mlir_tests = []

    test_files = _get_test_files_from_dir(directory)
    for filename in test_files:
        if not re.match(filter, filename):
            continue
        if filename not in EXCLUDE_MLIR_TESTS and _is_gpu_test_supported(
            gpu_arch, filename
        ):
            mlir_tests.append(os.path.join(directory, filename))

    results = []
    for test in mlir_tests:
        results.append(compile_and_run_mlir(test, target, verbose, mode))
    return results


def run_mlir_cpu_test(filter, verbose):
    from execute import compile_and_run_mlir
    directory = os.path.join(CUR_DIR, "mlir_tests", "cpu_ops")
    cpu_target = "cpu"
    mlir_tests = []

    test_files = _get_test_files_from_dir(directory)
    for filename in test_files:
        if not re.match(filter, filename):
            continue
        if filename not in EXCLUDE_MLIR_CPU_TESTS:
            mlir_tests.append(
                [
                    os.path.join(directory, filename),
                    MLIR_CPU_SPECIAL_INPUTS[filename]
                    if filename in MLIR_CPU_SPECIAL_INPUTS
                    else None,
                ]
            )

    results = []
    for test in mlir_tests:
        if test[1] is None:
            results.append(compile_and_run_mlir(test[0], cpu_target, verbose))
        else:
            results.append(
                compile_and_run_mlir(
                    test[0],
                    cpu_target,
                    random_mode=test[1][0],
                    low=test[1][1],
                    high=test[1][2],
                )
            )

    return results


def run_torch_test(target, gpu_arch, filter, verbose, mode):
    from execute import compile_and_run_torch
    tests = [
        test
        for test in GLOBAL_TORCH_TEST_REGISTRY
        if re.match(filter, test.unique_name)
        and test.unique_name not in EXCLUDE_TORCH_TESTS
        and _is_gpu_test_supported(gpu_arch, test.unique_name)
    ]
    results = []
    for test in tests:
        results.append(compile_and_run_torch(test, target, verbose, mode))
    return results


def run(config, target, filter, mode="numerical", verbose=False):
    if config == "mlir" and target == "cpu":
        return run_mlir_cpu_test(filter, verbose)

    from byteir.utils import detect_gpu_arch_with_nvidia_smi
    gpu_arch = detect_gpu_arch_with_nvidia_smi()
    assert gpu_arch != None
    assert gpu_arch.startswith("sm_")
    gpu_arch = int(gpu_arch[3:])
    if config == "mlir":
        return run_mlir_test(target, gpu_arch, filter, verbose, mode)
    elif config == "torch":
        return run_torch_test(target, gpu_arch, filter, verbose, mode)
    elif config == "dynamo":
        from torch_dynamo_e2e_testing.execute import run_torch_dynamo_tests
        # TODO(zzk): use test infra for dynamo tests
        run_torch_dynamo_tests(gpu_arch)
        return []
    assert False, f"unknown config: {config}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        choices=["all", "mlir", "torch", "dynamo"],
        required=True,
        help="test sets to run.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="cuda_with_ait",
        choices=[
            "cpu",
            "cuda",
            "cuda_with_ait",
            "cuda_with_ait_aggressive",
            "native_torch",
        ],
        help="target backend",
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
        help="report test results with additional detail",
    )
    args = parser.parse_args()
    return args


ALL_CONFIG = {
    "mlir": "cpu",
    "mlir": "cuda_with_ait",
    "torch": "cuda_with_ait",
    "dynamo": None,
}


def main():
    args = parse_args()

    results = []
    if args.config == "all":
        for config, target in ALL_CONFIG.items():
            results += run(config, target, args.filter)
    else:
        results += run(args.config, args.target, args.filter, mode=args.mode, verbose=args.verbose)

    failed = report_results(results)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    register_all_torch_tests()
    main()
