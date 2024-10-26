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
prepare_for_build

pushd $PROJ_DIR
cmake -S . \
      -B ./build \
      -GNinja \
      -DLLVM_EXTERNAL_LIT=$(which lit) \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang-11 \
      -DCMAKE_CXX_COMPILER=clang++-11 \
      -DTORCH_FRONTEND_ENABLE_JIT_IR_IMPORTER=${TORCH_FRONTEND_ENABLE_JIT_IR_IMPORTER} \
      -DCMAKE_CXX_FLAGS="-fPIC" \
      -DPython3_EXECUTABLE=$(which python3)

cmake --build ./build --target all

if [[ $TORCH_FRONTEND_TEST == "ON" ]]; then
  python3 -m pip install -r test-requirements.txt
  install_mhlo_tools
  PYTHONPATH=build/python_packages/:build/torch_mlir_build/python_packages/torch_mlir TORCH_DISABLE_NATIVE_FUNCOL=1 python3 -m pytest -m "not attention_rewriter" torch-frontend/python/test
fi

popd
