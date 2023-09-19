#!/bin/bash

set -e

CDIR="$(cd "$(dirname "$0")" ; pwd -P)"
ROOT_PROJ_DIR=$CDIR/..

pushd $ROOT_PROJ_DIR
git submodule update --init --recursive -f external/mlir-hlo external/AITemplate

function apply_mhlo_patches() {
  pushd $ROOT_PROJ_DIR/external/mlir-hlo
  git clean -fd .
  for patch in $ROOT_PROJ_DIR/external/patches/mlir-hlo/*; do
    git apply $patch
  done
  popd
}

function apply_aitemplate_patches() {
  pushd $ROOT_PROJ_DIR/external/AITemplate
  git clean -fd .
  for patch in $ROOT_PROJ_DIR/external/patches/AITemplate/*; do
    git apply $patch
  done
  popd
}

# note: need to apply mhlo patch with gcc8.3
# apply_mhlo_patches

apply_aitemplate_patches

popd
