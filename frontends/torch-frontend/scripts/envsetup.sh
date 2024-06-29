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
  TORCH_FRONTEND_LLVM_INSTALL_DIR="/data00/llvm_libraries/d16b21b17d13ecd88a068bb803df43e53d3b04ba/llvm_build"
}

function apply_patches() {
  pushd $TORCH_MLIR_ROOT
  git clean -fd .
  for patch in ../patches/*; do
    git apply $patch
  done
  popd
}

function prepare_for_build_with_prebuilt() {
  pushd ${PROJ_DIR}
  # install requirements
  python3 -m pip install -r requirements.txt -r torch-requirements.txt
  python3 -m pip install /data00/mhlo_libraries/mhlo_tools-1.3.0-cp39-cp39-linux_x86_64.whl
  python3 -m pip install --no-cache-dir torch==2.1.0+cu118 torchvision==0.16.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html

  # initialize submodule
  git submodule update --init -f $TORCH_MLIR_ROOT
  pushd $TORCH_MLIR_ROOT
  git submodule update --init -f externals/stablehlo
  popd

  apply_patches
  load_pytorch_llvm_prebuilt
}

function prepare_for_build() {
  pushd ${PROJ_DIR}
  # install requirements
  python3 -m pip install -r requirements.txt -r torch-requirements.txt
  python3 -m pip install /data00/mhlo_libraries/mhlo_tools-1.3.0-cp39-cp39-linux_x86_64.whl
  python3 -m pip install --no-cache-dir torch==2.1.0+cu118 torchvision==0.16.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html

  # initialize submodule
  git submodule update --init --recursive -f $TORCH_MLIR_ROOT

  apply_patches
}
