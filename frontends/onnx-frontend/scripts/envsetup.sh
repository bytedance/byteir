#!/bin/bash

function of_envsetup() {
  pushd $ONNX_FRONTEND_ROOT

  # install requirements
  python3 -m pip install -r $ONNX_FRONTEND_ROOT/requirements.txt

  # init submodule
  ONNX_MLIR_ROOT=$ONNX_FRONTEND_ROOT/third_party/onnx-mlir
  ONNX_OFFICIAL_ROOT=$ONNX_MLIR_ROOT/third_party/onnx

  git submodule update --init --recursive $ONNX_MLIR_ROOT
  git submodule update -f $ONNX_MLIR_ROOT
  pushd $ONNX_MLIR_ROOT
  git clean -fd .
  git apply $ONNX_FRONTEND_ROOT/third_party/patches/OnnxMlir*.patch
  git submodule update -f $ONNX_OFFICIAL_ROOT
  popd

  pushd $ONNX_OFFICIAL_ROOT
  git apply $ONNX_FRONTEND_ROOT/third_party/patches/OnnxOfficial*.patch
  popd

  popd
}

function of_build() {
  if [ ! -d ${ONNX_FRONTEND_ROOT}/build ]; then
    mkdir ${ONNX_FRONTEND_ROOT}/build
  fi

  cmake "-H$ONNX_FRONTEND_ROOT" \
      "-B$ONNX_FRONTEND_ROOT/build" \
      -GNinja \
      -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
      -DPython3_ROOT_DIR=/usr/bin/python3.9 \
      -DPY_VERSION=3 \
      -DMLIR_DIR="$ONNX_FRONTEND_LLVM_RTTI_INSTALL_DIR/lib/cmake/mlir" \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_EXTERNAL_LIT=$(which lit)

  cmake --build "$ONNX_FRONTEND_ROOT/build" --config Release --target onnx-frontend onnx-frontend-opt
}

function of_test_lit() {
  cmake --build "$ONNX_FRONTEND_ROOT/build" --target check-of-lit
  cmake --build "$ONNX_FRONTEND_ROOT/build" --target check-onnx-lit
}

function of_test_ops() {
  pushd $ONNX_FRONTEND_ROOT
  python3 -m pytest $ONNX_FRONTEND_ROOT/test/ops -s
  python3 -m pytest $ONNX_FRONTEND_ROOT/test/models -s
  popd
}

function of_format() {
  find $ONNX_FRONTEND_ROOT/onnx-frontend/ -iname *.h -o -iname *.cpp | xargs clang-format-13 -i -style=file
}
