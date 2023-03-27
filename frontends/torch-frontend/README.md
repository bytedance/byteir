# Torch Frontend
torch-frontend is a project to build customized torch model --> torch dialect --> mhlo dialect pipeline, where we could add extended dialect and passes.


## Quick Start

### Build from source code

```
git clone https://github.com/bytedance/byteir.git
cd byteir

# prepare python environment and torch-mlir dependency
bash ./frontends/torch-frontend/scripts/prepare.sh

cd frontends/torch-frontend

cmake -S . \
      -B ./build \
      -GNinja \
      -DLLVM_EXTERNAL_LIT=$(which lit) \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++

cmake --build ./build --target all
```

### Example
```
PYTHONPATH=./build/python_packages/ python3 examples/infer_resnet.py
```
