#!/bin/bash

set -e

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# path to byteir/frontends/torch-frontend
TORCH_FRONTEND_DIR="$CUR_DIR/.."
# path to torch-mlir
TORCH_MLIR_DIR="$TORCH_FRONTEND_DIR/third_party/torch-mlir"
# path to torch-mlir patches
TORCH_MLIR_PATCHES_DIR="$TORCH_FRONTEND_DIR/third_party/patches"

pushd $TORCH_FRONTEND_DIR
# install requirements
python3 -m pip install -r requirements.txt

# init submodule
# git submodule update --init --recursive -f $TORCH_MLIR_DIR

# apply patches
pushd $TORCH_MLIR_DIR
git clean -fd .
for patch in $TORCH_MLIR_PATCHES_DIR/*; do
  git apply $patch
done
popd

popd
