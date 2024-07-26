#!/bin/bash

set -e
set -x

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

pushd $CUR_DIR

# CUDA
# python3 gen_testcases.py --top-dir ../test/E2E/CUDA/MLPInference --category=E2E
python3 gen_testcases.py --top-dir ../test/E2E/CUDA/AliasLikeGPU --category=E2E
# TODO: add more CUDA E2E checker

# Host
python3 gen_testcases.py --top-dir ../test/E2E/Host/Case0 --category HostPipeline
python3 gen_testcases.py --top-dir ../test/E2E/Host/Case1 --category HostPipeline
python3 gen_testcases.py --top-dir ../test/E2E/Host/RngNormal --category HostPipeline
python3 gen_testcases.py --top-dir ../test/E2E/Host/RngUniform --category HostPipeline
python3 gen_testcases.py --top-dir ../test/E2E/Host/Transpose --category HostPipeline
python3 gen_testcases.py --top-dir ../test/E2E/Host/TypeCvt --category HostPipeline
python3 gen_testcases.py --top-dir ../test/E2E/Host/AliasLike --category HostPipeline
# Host Bytecode
python3 gen_testcases.py --top-dir ../test/E2E/Host/Case0_Bytecode --category HostPipelineBytecode

git diff --quiet ../test/E2E

popd
