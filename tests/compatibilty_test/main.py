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
import sys

from reporting import report_results

from execute import run_and_check_mlir

"""
Usage:
    This directory implements the code for compatibilty test framework. One should pass a test dir which contains:
    (1) subdirs for each tese case
    (2) byre compilation artifacts named as {model_name}/{model_name}.rt.mlir
    (3) several inputs and goldens named as inputs.{num}.npy and outputs.{num}.npy
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=str,
        default="all",
        choices=[
            "all",
            "cpu",
        ],
        help="target backend to run",
    )
    parser.add_argument(
        "--testdir",
        type=str,
        default=None,
        help="Directory has test cases",
    )
    args = parser.parse_args()
    return args


def run(target, testdir):

    def extract_name_from_tesrdir(testdir):
        return os.path.basename(testdir)

    result = []
    if target == "cpu":
        for subdir in os.listdir(testdir):
            casedir = os.path.join(testdir, subdir)
            result += run_and_check_mlir(extract_name_from_tesrdir(casedir),
                                         casedir, "cpu")
    return result


def main():
    args = parse_args()

    results = []
    if args.target == "all":
        for target in ["cpu"]:
            results += run(target, args.testdir)
    else:
        results += run(args.target, args.testdir)

    failed = report_results(results)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
