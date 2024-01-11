#!/bin/bash

set -e
set -x

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# path to byteir/frontends/tf-frontend
PROJ_DIR="$CUR_DIR/.."

pushd $PROJ_DIR

git submodule update --init --recursive -f $PROJ_DIR/external/tensorflow

# configure bazel
BAZEL_VERSION=$(head -n 1 $PROJ_DIR/external/tensorflow/.bazelversion)
if [ ! -f "bazel-$BAZEL_VERSION-linux-x86_64" ]; then
  wget https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-linux-x86_64 -q
fi
cp bazel-$BAZEL_VERSION-linux-x86_64 $PROJ_DIR/bazel -f
chmod +x $PROJ_DIR/bazel
cp $PROJ_DIR/.tf_configure.bazelrc $PROJ_DIR/external/tensorflow/.tf_configure.bazelrc

# apply patches
bash $CUR_DIR/apply_patches.sh

popd