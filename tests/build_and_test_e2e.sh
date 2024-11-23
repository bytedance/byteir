#!/bin/bash

set -e
set -x

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# path to byteir root
ROOT_PROJ_DIR="$CUR_DIR/.."

pushd $ROOT_PROJ_DIR
# build compiler
bash scripts/compiler/build_and_test.sh --no-test
# build runtime
bash scripts/runtime/build_and_test.sh --cuda --python --no-test
# build torch_frontend
pip3 install -r $ROOT_PROJ_DIR/frontends/torch-frontend/torch-cuda-requirements.txt
bash frontends/torch-frontend/scripts/build_and_test.sh --no-test

pip3 install $ROOT_PROJ_DIR/external/AITemplate/python/dist/*.whl
pip3 install $ROOT_PROJ_DIR/compiler/build/python/dist/*.whl
pip3 install $ROOT_PROJ_DIR/runtime/python/dist/*.whl
pip3 install $ROOT_PROJ_DIR/frontends/torch-frontend/build/torch-frontend/python/dist/*.whl
source scripts/prepare.sh
install_mhlo_tools

# numerical test
# pip3 install flash_attn==2.5.3
python3 tests/numerical_test/main.py --target all
rm -rf ./local_test

# profiler test
python3 tests/numerical_test/profiler.py $ROOT_PROJ_DIR/tests/numerical_test/mlir_tests/cpu_ops/add.mlir --target cpu
python3 tests/numerical_test/profiler.py $ROOT_PROJ_DIR/tests/numerical_test/mlir_tests/ops/add.mlir --target cuda
rm -rf ./local_profiling

# generate and run compitibility test
python3 tests/numerical_test/gen_brt_tests.py
python3 tests/compatibility_test/main.py --testdir=./local_golden
rm -rf ./local_golden

popd
