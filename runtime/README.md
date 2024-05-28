# ByteIR Runtime

ByteIR Runtime is a common Runtime mainly serving both existing kernels and ByteIR compiler generated kernels.


## Dependency 

***LLVM/MLIR*** https://github.com/llvm/llvm-project, llvm commit id see `external/llvm-project`.

***ByteIR ByRE dialect***  https://github.com/bytedance/byteir/compiler

## Build and Run
### Build LLVM
build llvm like [ByteIR compiler](../compiler/README.md) does
### Build BRT(ByteIR Runtime)
#### Linux/Mac
```bash
cd /path_to_byteir

git submodule update --init --recursive -f external/mlir-hlo external/cutlass external/date external/googletest external/pybind11

# build runtime
cmake -H./runtime/cmake \
      -B./runtime/build \
      -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_INSTALL_PATH=$(pwd)/external/llvm-project/build/install \
      -DCMAKE_INSTALL_PREFIX="$(pwd)/runtime/build/install" \
      -Dbrt_ENABLE_PYTHON_BINDINGS=ON \
      -Dbrt_USE_CUDA=ON

cmake --build ./runtime/build --target all --target install
```

#### Windows
```bash
cd /path_to_byteir

git submodule update --init --recursive -f external/mlir-hlo external/cutlass external/date external/googletest external/pybind11

# build runtime
cmake -H./runtime/cmake \
      -B./runtime/build \
      -G "Visual Studio 16 2019" -A x64 \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_INSTALL_PATH=path_to_LLVM_installed_or_built_directory \
      -DCMAKE_INSTALL_PREFIX="$(pwd)/runtime/build/install" \
      -Dbrt_ENABLE_PYTHON_BINDINGS=ON \
      -Dbrt_USE_CUDA=ON

cmake --build ./runtime/build --target all --target install
```

### Pack Python Wheel (depend on build and install)
```bash
cd /path_to_runtime/python

python3 setup.py bdist_wheel
# brt-*.whl in /path_to_byteir/runtime/python/dist/
```

### Testing
#### Linux/Max
```bash
cd ./runtime/build
./bin/brt_test_all
# using --gtest_filter="*" to filter unit test by regular expression
```

### Run
See example like [add2.py](./python/examples/add2.py).
