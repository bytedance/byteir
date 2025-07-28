#!/bin/bash

set -e
set -x

TRITON_BUILD=ON
TRITON_TEST=ON

while [[ $# -gt 0 ]]; do
  case $1 in
    --no-test)
      TRITON_TEST=OFF
      shift
      ;;
    --no-build)
      TRITON_BUILD=OFF
      shift
      ;;
    *)
      echo "Invalid option: $1"
      exit 1
      ;;
  esac
done


TRITON_TEST=${TRITON_TEST:-ON}

# path to script
CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# path to TritonTemplate root
ROOT_PROJ_DIR="$CUR_DIR/.."

# path to python
PYTHON_PATH="$ROOT_PROJ_DIR/python"

if [[ $TRITON_BUILD == "ON" ]]; then
    pushd "$PYTHON_PATH"
        # install tritontemplate
        python3 -m pip install -e .
    popd
fi

# path to python tests
TEST_DIR="$ROOT_PROJ_DIR/python/tritontemplate/testing/cuda"

if [[ $TRITON_TEST == "ON" ]]; then
  # Run pytest for all test files in cuda directory
  pushd "$TEST_DIR"
  for file in *.py; do
      if [ -f "$file" ]; then
          echo "Running pytest on $file"
          GITHUB_CI_TEST=true pytest "$file"
      fi
  done
  popd
fi
