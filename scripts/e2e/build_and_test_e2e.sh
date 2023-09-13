#!/bin/bash

set -e
set -x

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# path to byteir root
ROOT_PROJ_DIR="$CUR_DIR/../.."

LLVM_INSTALL_DIR="$1"

TORCH_FRONTEND_LLVM_INSTALL_DIR="$2"

pushd $ROOT_PROJ_DIR
# build compiler
bash scripts/compiler/build_and_lit_test.sh $LLVM_INSTALL_DIR
# build runtime
bash scripts/runtime/build_and_test.sh --python --no-test $LLVM_INSTALL_DIR
# build torch_frontend
bash scripts/frontends/torch-frontend/build_and_test.sh $TORCH_FRONTEND_LLVM_INSTALL_DIR

pip3 install $ROOT_PROJ_DIR/external/AITemplate/python/dist/*.whl 
pip3 install $ROOT_PROJ_DIR/compiler/build/python/dist/*.whl
pip3 install $ROOT_PROJ_DIR/runtime/python/dist/*.whl
pip3 install $ROOT_PROJ_DIR/frontends/torch-frontend/build/torch-frontend/python/dist/*.whl 
pip3 install -r $ROOT_PROJ_DIR/frontends/torch-frontend/torch-requirements.txt

python3 tests/numerical_test/main.py
rm -rf ./local_test
popd
