# Torch Frontend
torch-frontend is a project to build customized torch model --> torch dialect --> mhlo dialect pipeline, where we could add extended dialect and passes.


## Quick Start

### Build from source code

```bash
git clone https://github.com/bytedance/byteir.git
cd byteir/frontends/torch-frontend

# prepare python environment and torch-mlir dependency
bash scripts/prepare.sh

cmake -S . \
      -B ./build \
      -GNinja \
      -DLLVM_EXTERNAL_LIT=$(which lit) \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DPython3_EXECUTABLE=$(which python3)

cmake --build ./build --target all
# torch_frontend-*.whl in ./build/torch-frontend/python/dist/
```

### Example
```bash
PYTHONPATH=./build/python_packages/ python3 examples/inference/infer_resnet.py
```
