#!/bin/bash

set -e

CUR_DIR="$(cd "$(dirname "$0")" ; pwd -P)"
TF_FRONTEND_DIR=$CUR_DIR/..
TF_DIR=$TF_FRONTEND_DIR/external/tensorflow
TF_PATCHES_DIR=$TF_FRONTEND_DIR/external/patches/tensorflow

pushd $TF_DIR
git clean -fd .
for patch in $TF_PATCHES_DIR/*; do
  git apply $patch
done
popd
