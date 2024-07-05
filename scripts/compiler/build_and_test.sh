#!/bin/bash

set -e
set -x

while [[ $# -gt 0 ]]; do
  case $1 in
    --no-test)
      BYTEIR_TEST=OFF
      shift
      ;;
    *)
      echo "Invalid option: $1"
      exit 1
      ;;
  esac
done

BYTEIR_TEST=${BYTEIR_TEST:-ON}

# path to script
CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# path to byteir root
ROOT_PROJ_DIR="$CUR_DIR/../.."
# path to byteir/compiler
PROJ_DIR="$ROOT_PROJ_DIR/compiler"
# dir to build
BUILD_DIR="$PROJ_DIR/build"
# dir to install
INSTALL_DIR="$BUILD_DIR/byre_install"

source $CUR_DIR/../prepare.sh
prepare_for_compiler

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cmake "-H$PROJ_DIR/cmake" \
      "-B$BUILD_DIR" \
      -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_INSTALL_PATH="$LLVM_INSTALL_DIR" \
      -DLLVM_EXTERNAL_LIT=$(which lit) \
      -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
      -DCMAKE_CXX_FLAGS="-Wno-unused-but-set-parameter -Wno-unused-but-set-variable -Wno-maybe-uninitialized" \
      -DBYTEIR_ENABLE_BINDINGS_PYTHON=ON

cmake --build "$BUILD_DIR" --target all check-byteir install
cmake --build "$BUILD_DIR" --target byteir-python-pack

if [[ $BYTEIR_TEST == "ON" ]]; then
  install_mhlo_tools
  # test numerical
  cmake --build "$BUILD_DIR" --target check-byteir-numerical
  # test cat
  cmake --build "$BUILD_DIR" --target check-byteir-python
  # pytest
  pushd $ROOT_PROJ_DIR
  PYTHONPATH=./compiler/build/python_packages/byteir python3 -m pytest ./compiler/python/test/api
  popd
fi
