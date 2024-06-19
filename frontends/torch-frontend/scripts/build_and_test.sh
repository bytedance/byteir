#!/bin/bash

set -e
set -x

while [[ $# -gt 0 ]]; do
  case $1 in
    --disable-jit-ir)
      TORCH_FRONTEND_ENABLE_JIT_IR_IMPORTER=OFF
      shift
      ;;
    --no-test)
      TORCH_FRONTEND_TEST=OFF
      shift
      ;;
    *)
      echo "Invalid option: $1"
      exit 1
      ;;
  esac
done

TORCH_FRONTEND_ENABLE_JIT_IR_IMPORTER=${TORCH_FRONTEND_ENABLE_JIT_IR_IMPORTER:-ON}
TORCH_FRONTEND_TEST=${TORCH_FRONTEND_TEST:-ON}

# path to script
CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# path to byteir root
ROOT_PROJ_DIR="$CUR_DIR/../../.."
# path to byteir/frontends/torch-frontend
PROJ_DIR="$ROOT_PROJ_DIR/frontends/torch-frontend"

source $CUR_DIR/envsetup.sh
prepare_for_build_with_prebuilt

pushd $PROJ_DIR
cmake -S . \
      -B ./build \
      -GNinja \
      -DMLIR_DIR="$TORCH_FRONTEND_LLVM_INSTALL_DIR/lib/cmake/mlir" \
      -DLLVM_EXTERNAL_LIT=$(which lit) \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=gcc \
      -DCMAKE_CXX_COMPILER=g++ \
      -DTORCH_FRONTEND_ENABLE_JIT_IR_IMPORTER=${TORCH_FRONTEND_ENABLE_JIT_IR_IMPORTER} \
      -DCMAKE_CXX_FLAGS="-Wno-unused-but-set-parameter -Wno-unused-but-set-variable" \
      -DPython3_EXECUTABLE=$(which python3)

cmake --build ./build --target all

if [[ $TORCH_FRONTEND_TEST == "ON" ]]; then
  PYTHONPATH=build/python_packages/:build/torch_mlir_build/python_packages/torch_mlir python3 -m pytest torch-frontend/python/test
fi

popd
