#!/bin/bash

set -e
set -x

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# path to byteir root
ROOT_PROJ_DIR="$CUR_DIR/.."

pushd $ROOT_PROJ_DIR
# build compiler
bash scripts/compiler/build_and_test.sh --no-test
# build runtime
bash scripts/runtime/build_and_test.sh --cuda --python --no-test
# build torch_frontend
bash frontends/torch-frontend/scripts/build_and_test.sh --no-test

pip3 install $ROOT_PROJ_DIR/external/AITemplate/python/dist/*.whl
pip3 install $ROOT_PROJ_DIR/compiler/build/python/dist/*.whl --force-reinstall
pip3 install $ROOT_PROJ_DIR/runtime/python/dist/*.whl --force-reinstall
pip3 install $ROOT_PROJ_DIR/frontends/torch-frontend/build/torch-frontend/python/dist/*.whl --force-reinstall
pip3 install -r $ROOT_PROJ_DIR/frontends/torch-frontend/torch-requirements.txt
pip3 install flash_attn==2.5.3
python3 tests/numerical_test/main.py --target all
rm -rf ./local_test
popd
