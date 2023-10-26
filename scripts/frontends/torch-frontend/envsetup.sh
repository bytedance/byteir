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
  python3 -m pip install -r requirements.txt
  python3 -m pip install torch==2.1.0+cpu torchvision==0.16.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

  # init submodule
  git submodule update --init -f $TORCH_MLIR_ROOT
  pushd $TORCH_MLIR_ROOT
  git submodule update --init -f externals/stablehlo
  git submodule update --init -f externals/llvm-project
  popd

  # apply patches
  apply_patches
}
