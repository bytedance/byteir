#!/bin/bash

set -e
set -x

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

pushd $CUR_DIR

# CUDA
# TODO: add CUDA E2E checker when pipeline fixed

# Host
python3 gen_testcases.py --top-dir ../test/E2E/Host/Case0 --category HostPipeline
python3 gen_testcases.py --top-dir ../test/E2E/Host/Case1 --category HostPipeline
python3 gen_testcases.py --top-dir ../test/E2E/Host/RngNormal --category HostPipeline
python3 gen_testcases.py --top-dir ../test/E2E/Host/RngUniform --category HostPipeline
python3 gen_testcases.py --top-dir ../test/E2E/Host/Transpose --category HostPipeline
python3 gen_testcases.py --top-dir ../test/E2E/Host/TypeCvt --category HostPipeline
# Host Bytecode
python3 gen_testcases.py --top-dir ../test/E2E/Host/Case0_Bytecode --category HostPipelineBytecode

# git diff --quiet ../test

popd
