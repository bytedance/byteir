#!/bin/bash

set -e

while [[ $# -gt 1 ]]; do
  case $1 in
    --cuda)
      BRT_USE_CUDA=ON
      shift
      ;;
    --asan)
      BRT_ENABLE_ASAN=ON
      CMAKE_BUILD_TYPE=Debug
      shift
      ;;
    --python)
      BRT_ENABLE_PYTHON_BINDINGS=ON
      shift
      ;;
    --no-test)
      BRT_TEST=OFF
      shift
      ;;
    --flash)
      BRT_ENABLE_FLASH_ATTENSION=ON
      shift
      ;;
    *)
      echo "Invalid option: $1"
      exit 1
      ;;
  esac
done

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# path to byteir root
ROOT_PROJ_DIR="$CUR_DIR/../.."
# path to byteir/runtime
PROJ_DIR="$ROOT_PROJ_DIR/runtime"

# dir to build
BUILD_DIR="$PROJ_DIR/build"
# dir to install
INSTALL_DIR="$BUILD_DIR/install"

# build options
BRT_USE_CUDA=${BRT_USE_CUDA:-OFF}
BRT_ENABLE_ASAN=${BRT_ENABLE_ASAN:-OFF}
BRT_ENABLE_PYTHON_BINDINGS=${BRT_ENABLE_PYTHON_BINDINGS:-OFF}
CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release}
BRT_ENABLE_FLASH_ATTENSION=${BRT_ENABLE_FLASH_ATTENSION:-OFF}
# test options
BRT_TEST=${BRT_TEST:-ON}

source $CUR_DIR/../prepare.sh
prepare_for_runtime
LLVM_INSTALL_DIR="$1"

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cmake -GNinja \
  "-H$PROJ_DIR/cmake" \
  "-B$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE \
  -DLLVM_INSTALL_PATH="$LLVM_INSTALL_DIR" \
  -DCMAKE_INSTALL_PREFIX="$BUILD_DIR/install" \
  -Dbrt_USE_CUDA=${BRT_USE_CUDA} \
  -Dbrt_ENABLE_FLASH_ATTENSION=${BRT_ENABLE_FLASH_ATTENSION} \
  -Dbrt_ENABLE_ASAN=${BRT_ENABLE_ASAN} \
  -Dbrt_ENABLE_PYTHON_BINDINGS=${BRT_ENABLE_PYTHON_BINDINGS}

cmake --build "$BUILD_DIR" --target all --target install

if [[ $BRT_ENABLE_PYTHON_BINDINGS == "ON" ]]; then
  pushd $PROJ_DIR/python
  # note: python packing depend on `--target install`
  python3 setup.py bdist_wheel
  popd
fi

if [[ $BRT_USE_CUDA == "ON" ]] && [[ $BRT_ENABLE_ASAN == "ON" ]]; then
  export ASAN_OPTIONS=protect_shadow_gap=0
fi

if [[ $BRT_TEST == "ON" ]]; then
  pushd $BUILD_DIR
  ./bin/brt_test_all
  popd
fi
