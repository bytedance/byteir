#!/bin/bash

set -x
set -e

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"
# path to byteir root
BYTEIR_ROOT="$CUR_DIR/../../.."
# path to byteir/frontends/onnx-frontend
ONNX_FRONTEND_ROOT="$BYTEIR_ROOT/frontends/onnx-frontend"

export BYTEIR_ROOT="$BYTEIR_ROOT"
export ONNX_FRONTEND_ROOT="$ONNX_FRONTEND_ROOT"

source $CUR_DIR/envsetup.sh
source $BYTEIR_ROOT/scripts/prepare.sh
load_onnx_llvm_rtti_prebuilt

of_envsetup
of_build
of_test_lit
of_test_ops
