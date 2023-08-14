#!/bin/bash

set -e

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJ_DIR="$CUR_DIR/../../runtime"
BRT_INSTALL_DIR="$PROJ_DIR/build/install"
EXTERNAL_PROJECT_SRC_DIR="$PROJ_DIR/examples/external_project"
EXTERNAL_PROJECT_BUILD_DIR="$EXTERNAL_PROJECT_SRC_DIR/build"

rm -rf "$EXTERNAL_PROJECT_BUILD_DIR"
mkdir -p "$EXTERNAL_PROJECT_BUILD_DIR"
cmake -GNinja \
  "-H$EXTERNAL_PROJECT_SRC_DIR" \
  "-B$EXTERNAL_PROJECT_BUILD_DIR" \
  -DBRT_INSTALL_PATH="$BRT_INSTALL_DIR"

cmake --build "$EXTERNAL_PROJECT_BUILD_DIR" --target all
pushd $EXTERNAL_PROJECT_BUILD_DIR
./main
popd
