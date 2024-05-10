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
from execute import gen_golden_mlir
from utils import report_results
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--target",
                    type=str,
                    default="cpu",
                    choices=["all", "cuda", "cpu"],
                    help="target device name")
parser.add_argument("-g",
                    "--golden",
                    default="/tmp/mlir_cpu_golden",
                    help="mlir test golden path")
args = parser.parse_args()

EXCLUDE_MLIR_TESTS = []

# Unsupported ops
EXCLUDE_MLIR_CPU_TESTS = [
    "custom_call_tf_UpperBound.mlir",
    "rng.mlir",
]

MLIR_CPU_SPECIAL_INPUTS = {
    "log_plus_one.mlir": ["uniform", 0.5, 1.0],
}


def gen_mlir_cpu_golden():
    directory = os.path.dirname(os.path.realpath(__file__))
    directory = directory + "/mlir_tests/cpu_ops"
    cpu_target = "cpu"
    mlir_tests = []
    os.makedirs(args.golden, exist_ok=True)

    for filename in os.listdir(directory):
        if filename.startswith('.'):
            continue
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f) and filename not in EXCLUDE_MLIR_CPU_TESTS:
            mlir_tests.append([
                f, MLIR_CPU_SPECIAL_INPUTS[filename]
                if filename in MLIR_CPU_SPECIAL_INPUTS else None
            ])

    results = []
    for test in mlir_tests:
        fpath = test[0]
        cur_golden_dir = args.golden
        if test[1] is None:
            res = gen_golden_mlir(fpath,
                                  cpu_target,
                                  golden_dir=cur_golden_dir,
                                  num=5)
        else:
            res = gen_golden_mlir(fpath,
                                  cpu_target,
                                  golden_dir=cur_golden_dir,
                                  num=5,
                                  mode=test[1][0],
                                  low=test[1][1],
                                  high=test[1][2])

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
