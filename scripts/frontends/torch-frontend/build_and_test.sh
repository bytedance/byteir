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
      -DPython3_EXECUTABLE=$(which python3)

cmake --build ./build --target all

python3 -m pip install -r ./torch-requirements.txt
PYTHONPATH=./build/python_packages/ python3 -m pytest torch-frontend/python/test

popd
