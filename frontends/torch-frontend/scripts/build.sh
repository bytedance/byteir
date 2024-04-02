#!/bin/bash

set -e
set -x

# path to script
CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# path to byteir root
ROOT_PROJ_DIR="$CUR_DIR/../../.."
# path to byteir/frontends/torch-frontend
PROJ_DIR="$ROOT_PROJ_DIR/frontends/torch-frontend"

source $CUR_DIR/envsetup.sh
prepare_for_build

pushd $PROJ_DIR
cmake -S . \
      -B ./build \
      -GNinja \
      -DLLVM_EXTERNAL_LIT=$(which lit) \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=gcc \
      -DCMAKE_CXX_COMPILER=g++ \
      -DCMAKE_CXX_FLAGS="-Wno-unused-but-set-parameter -Wno-unused-but-set-variable" \
      -DPython3_EXECUTABLE=$(which python3)

cmake --build ./build --target all

PYTHONPATH=build/python_packages/:build/torch_mlir_build/python_packages/torch_mlir python3 -m pytest torch-frontend/python/test

popd
