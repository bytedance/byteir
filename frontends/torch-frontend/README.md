# How to

## Build torch-frontend

1. git submodule init

    $ git submodule update --init --recursive third_party/torch-mlir

2. install requirements

    $ python3 -m pip install requirements.txt

3. configuration

    - configure torch-frontend, llvm in torch-mlir will be built at configure time

      $ cmake -S . -B ./build -GNinja -DLLVM_EXTERNAL_LIT=$(which lit)

    - [alternative] configure torch-frontend with prebuilt mlir

      $ cmake -S . -B ./build -GNinja -DMLIR_DIR=path/to/installed/mlir -DLLVM_EXTERNAL_LIT=$(which lit)

4. build & run test

    $ cmake --build ./build --target all

    $ PYTHONPATH=./build/python_packages/ python3 -m pytest torch-frontend/python/test
