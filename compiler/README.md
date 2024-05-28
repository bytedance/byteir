# The ByteIR Compiler

The ByteIR Compiler is an MLIR-based compiler for CPU/GPU/ASIC.

## Dependency 
***LLVM/MLIR***: https://github.com/llvm/llvm-project, llvm commit id see `external/llvm-project`.

***Python*** (for python binding): minimum version is 3.6, requiring numpy and pybind11 installed.

## Build and Run

### Build LLVM

```bash
cd /path_to_byteir
git submodule update --init external/llvm-project

# build llvm
cd external/llvm-project
cmake -H./llvm \
      -B./build \
      -GNinja \
      -DLLVM_ENABLE_PROJECTS=mlir \
      -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DLLVM_INSTALL_UTILS=ON \
      -DLLVM_CCACHE_BUILD=OFF \
      -DLLVM_ENABLE_TERMINFO=OFF \
      -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
      -DCMAKE_INSTALL_PREFIX=$(pwd)/build/install
# via -DCMAKE_C_COMPILER=gcc/clang and -DCMAKE_CXX_COMPILER=g++/clang++
# to specify gcc>=8.5 or clang>=7 

cmake --build ./build --target all --target install
```

### Build ByteIR
#### Linux/Mac 
```bash
cd /path_to_byteir

git submodule update --init external/mlir-hlo

# build ByteIR
cmake -B./compiler/build \
      -H./compiler/cmake \
      -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_INSTALL_PATH=$(pwd)/external/llvm-project/build/install \
      -DLLVM_EXTERNAL_LIT=$(which lit) \
      -DBYTEIR_ENABLE_BINDINGS_PYTHON=ON

cmake --build ./compiler/build --target check-byteir
```
#### Windows 
```bash
cd /path_to_byteir

git submodule update --init external/mlir-hlo

# build ByteIR
cmake -B./compiler/build
      -H./compiler/cmake
      -G "Visual Studio 16 2019" -A x64 \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_INSTALL_PATH=path_to_LLVM_installed_or_built_directory \
      -DLLVM_EXTERNAL_LIT=lit_location # this is optional for external lit 

cmake --build ./compiler/build --target check-byteir
```

### Testing 
This command runs ByteIR unit tests:
```bash
cmake --build ./compiler/build --target check-byteir
```
ByteIR relies on ```llvm-lit``` and ```FileCheck``` for testing.
For more information, you can refer to [this page](https://www.llvm.org/docs/CommandGuide/FileCheck.html)
All the tests are placed in the folder ```compiler/test```.

### Run with Python Binding
This command is an example showing that how to compile model by using ByteIR:
```bash
PYTHONPATH=./compiler/build/python_packages/byteir python3 -m byteir.tools.compiler ./compiler/test/E2E/MLPInference/input.mlir -o out.mlir --entry_func forward
# add -v (means verbose) to see detailed compiling pipeline 
```


### Install (Optional)
```bash
cmake --install ./compiler/build --prefix path_to_install_BYTEIR
```

### Pack Python Wheel (Optional)
```bash
cmake --build ./compiler/build --target byteir-python-pack
# byteir-*.whl in /path_to_byteir/compiler/build/python/dist/
```

## IRs (Dialects)
ByteIR Compiler mainly relies on MLIR public, builtin dialects. 
However, there are still some exceptions.

### MHLO 
An external dialect from https://github.com/tensorflow/mlir-hlo.
MHLO is the main dialect ByteIR compiler takes as the input IR.

There are several MHLO custom_call Ops used in ByteIR.
They are used in a contract between ByteIR compiler and ByteIR maintained frontends.
They are listed in [doc/byteir_mhlo_custom_call.md](doc/byteir_mhlo_custom_call.md)

### ACE & LACE
ACE is an internal dialect defined by ByteIR. 
It is a supplement to MHLO dialect and LACE is the corresponding part of LMHLO.

### CCL
CCL is an internal dialect defined by ByteIR.
It is a IR which represents communication operators in distribution.

### LinalgExt
LinalgExt is a dialect defined by ByteIR.
It is an extension of Linalg dialect, 
and it is meant to eventually be upstreamed to LLVM.

### ShapeExt
ShapeExt is a dialect defined by ByteIR.
It is an extension of Shape dialect, 
and it is meant to eventually be upstreamed to LLVM.

### ByRE (ByteDance Representation for Execution)
ByRE is a dialect defined by ByteIR.
It is a runtime IR and the major format for the ByteIR runtime. 

## Passes
Useful Pass Description [doc/passes.md](doc/passes.md)
