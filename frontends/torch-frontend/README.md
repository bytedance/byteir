# Torch Frontend
torch-frontend is a project to build customized torch model --> torch dialect --> mhlo dialect pipeline, where we could add extended dialect and passes.


## Quick Start

### Build from source code

```bash
git clone https://github.com/bytedance/byteir.git
cd byteir/frontends/torch-frontend

# prepare python environment and build torch-frontend
bash scripts/build.sh

# torch_frontend-*.whl in ./build/torch-frontend/python/dist/
```

### Example
```bash
PYTHONPATH=./build/python_packages/:build/torch_mlir_build/python_packages/torch_mlir python3 examples/inference/infer_resnet.py
```
