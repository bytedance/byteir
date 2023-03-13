#!/bin/bash

set -e

CDIR="$(cd "$(dirname "$0")" ; pwd -P)"
TF_FRONTEND_DIR=$CDIR/..
TF_DIR=$TF_FRONTEND_DIR/external/tensorflow
TF_PATCHES_DIR=$TF_FRONTEND_DIR/external/patches/tensorflow

pushd $TF_DIR
for patch in $TF_PATCHES_DIR/*; do
  git apply $patch
done
popd
