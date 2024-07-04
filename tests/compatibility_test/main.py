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
import json

from reporting import report_results

from execute import run_and_check_mlir
"""
Usage:
    This directory implements the code for compatibilty test framework. One should pass a test dir which contains:
    (1) subdirs for each tese case and json conf file named `testcase.json`
    (2) byre compilation artifacts named as {model_name}/{model_name}.rt.mlir
    (3) several inputs and goldens named as inputs.{num}.npz and outputs.{num}.npz
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--testdir",
        type=str,
        required=True,
        help="Directory has test cases",
    )
    args = parser.parse_args()
    return args


def run(testdir):
    result = []
    conf_file = os.path.join(testdir, "testcase.json")
    if not os.path.exists(conf_file):
        raise RuntimeError(f"test case config file {conf_file} not found")
    with open(conf_file, "r", encoding='utf-8') as f:
        json_conf = json.load(f)
    for target, data in json_conf.items():
        for byre_version, cases in data.items():
            for name, files in cases.items():
                input_files = files["golden_inputs"]
                input_files = [os.path.join(testdir, f) for f in input_files]
                golden_files = files["golden_outputs"]
                golden_files = [os.path.join(testdir, f) for f in golden_files]
                byre_file = files["brt_entry_file"]
                byre_file = os.path.join(testdir, byre_file)
                if len(input_files) != len(golden_files):
                    raise RuntimeError(
                        f"num of inouts({len(input_files)}) and goldens({len(golden_files)}) not eq in {name}"
                    )
                if not os.path.exists(byre_file):
                    raise RuntimeError(f"byre file{byre_file} not found")
                result += run_and_check_mlir(target, name, input_files,
                                             golden_files, byre_file)
    return result


def main():
    args = parse_args()

    results = run(args.testdir)

    failed = report_results(results)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
