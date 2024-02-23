# ByteIR Runtime

ByteIR Runtime is a common Runtime mainly serving both existing kernels and ByteIR compiler generated kernels.


## Dependency 

***LLVM/MLIR*** https://github.com/llvm/llvm-project, llvm commit id see `external/llvm-project`.

***ByteIR ByRE dialect***  https://github.com/bytedance/byteir/compiler

## Build
### Linux/Mac
```bash
mkdir /path_to_runtime/build
cd /path_to_runtime/build/

# build runtime
cmake ../cmake/ -G Ninja \
                -DCMAKE_BUILD_TYPE=Release \
                -DLLVM_INSTALL_PATH=path_to_LLVM_installed_or_built_directory \
                -DCMAKE_INSTALL_PREFIX="$(pwd)/install" \
                -Dbrt_ENABLE_PYTHON_BINDINGS=ON \
                -Dbrt_USE_CUDA=On

cmake --build . --target all --target install
```

### Windows
```bash
mkdir /path_to_runtime/build
cd /path_to_runtime/build/

# build runtime
cmake ..\cmake -G "Visual Studio 16 2019" -A x64 \
                -DCMAKE_BUILD_TYPE=Release \
                -DLLVM_INSTALL_PATH=path_to_LLVM_installed_or_built_directory \
                -DCMAKE_INSTALL_PREFIX="$(pwd)/install" \
                -Dbrt_ENABLE_PYTHON_BINDINGS=ON \
                -Dbrt_USE_CUDA=On

cmake --build . --target all --target install
```

## Pack python wheel (depend on build and install)
```bash
cd /path_to_runtime/python

python3 setup.py bdist_wheel
# brt-*.whl in /path_to_runtime/python/dist/
```

## Test your build
### Linux/Max
```bash
cd /path_to_runtime/build
./bin/brt_test_all
```

