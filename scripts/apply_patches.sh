#!/bin/bash

set -e

CDIR="$(cd "$(dirname "$0")" ; pwd -P)"
ROOT_PROJ_DIR=$CDIR/..

pushd $ROOT_PROJ_DIR
git submodule update --init --recursive -f external/mlir-hlo

pushd $ROOT_PROJ_DIR/external/mlir-hlo
git clean -fd .
for patch in $ROOT_PROJ_DIR/external/patches/mlir-hlo/*; do
  git apply $patch
done
popd

popd
