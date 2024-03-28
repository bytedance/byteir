#!/bin/bash

set -e
set -x

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# path to byteir/frontends/tf-frontend
PROJ_DIR="$CUR_DIR/.."

bash $PROJ_DIR/scripts/prepare.sh

pushd $PROJ_DIR
python3 -m pip install /data00/mhlo_libraries/mhlo_tools-1.3.0-cp39-cp39-linux_x86_64.whl
$PROJ_DIR/bazel --output_user_root=./build build //tools:tf-frontend //tools:tf-ext-opt
$PROJ_DIR/bazel --output_user_root=./build test --test_output=errors //tf_mlir_ext/tests:all --java_runtime_version=remotejdk_11
$PROJ_DIR/bazel --output_user_root=./build test --test_output=errors //tf_mlir_ext/numerical:all --java_runtime_version=remotejdk_11
popd
