# ByteIR Runtime

ByteIR Runtime is a common Runtime mainly serving both existing kernels and ByteIR compiler generated kernels.


## Dependency 

***LLVM/MLIR*** https://github.com/llvm/llvm-project.git

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
                (extra options, such as -Dbrt_USE_CUDA=On)

cmake --build . --config Release
```

### Windows
```bash
mkdir /path_to_runtime/build
cd /path_to_runtime/build/

# build runtime
cmake ..\cmake -G "Visual Studio 16 2019" -A x64 \
                -DCMAKE_BUILD_TYPE=Release \
                -DLLVM_INSTALL_PATH=path_to_LLVM_installed_or_built_directory \
                (extra options, such as -Dbrt_USE_CUDA=On)

cmake --build . --config Release
```

## Test your build
### Linux/Max
```bash
cd build
./bin/brt_test_all
```

