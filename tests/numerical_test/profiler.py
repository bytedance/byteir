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

import argparse
import sys

from execute import compile_and_run_mlir
from reporting import report_results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_mlir_path")
    parser.add_argument("--workdir", type=str, default="./profiling", help="workspace directory")
    parser.add_argument("--name", type=str, default="model")
    parser.add_argument("--target", type=str, default="cuda", choices=["cpu", "cuda", "cuda_with_ait"])
    parser.add_argument("--mode", type=str, default="profile", choices=["numerical", "profile"])
    parser.add_argument("-v", "--verbose", default=False, action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    result = compile_and_run_mlir(args.input_mlir_path, args.target, args.verbose, mode=args.mode, workdir=args.workdir, unique_name=args.name)
    failed = report_results([result])
    sys.exit(1 if failed else 0)
