#!/bin/bash

set -e
set -x

# path to script
CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# path to byteir root
ROOT_PROJ_DIR="$CUR_DIR/../../.."
# path to byteir/frontends/torch-frontend
PROJ_DIR="$ROOT_PROJ_DIR/frontends/torch-frontend"
# path to torch-mlir
TORCH_MLIR_ROOT="$PROJ_DIR/third_party/torch-mlir"

function load_pytorch_llvm_prebuilt() {
  TORCH_FRONTEND_LLVM_INSTALL_DIR="/data00/llvm_libraries/f7250179e22ce4aab96166493b27223fa28c2181/llvm_build"
}

function apply_patches() {
  pushd $TORCH_MLIR_ROOT
  git clean -fd .
  for patch in ../patches/*; do
    git apply $patch
  done
  popd
}

function prepare_for_build() {
  pushd ${PROJ_DIR}
  # install requirements
  python3 -m pip install -r requirements.txt -r torch-requirements.txt
  # python3 -m pip install torch==2.1.0+cpu torchvision==0.16.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

  # init submodule
  git submodule update --init -f $TORCH_MLIR_ROOT
  pushd $TORCH_MLIR_ROOT
  git submodule update --init -f externals/stablehlo
  popd

  # apply patches
  apply_patches
}
