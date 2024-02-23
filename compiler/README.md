# The ByteIR Compiler

The ByteIR Compiler is an MLIR-based compiler for CPU/GPU/ASIC.

## Dependency 
***LLVM/MLIR***: https://github.com/llvm/llvm-project, llvm commit id see `external/llvm-project`.

***Python*** (for python binding): minimum version is 3.6, requiring numpy and pybind11 installed.

## Build

### Apply Patches
Make sure to apply possible patches for submodules
```bash
bash /path_to_byteir/scripts/apply_patches.sh
```

### Linux/Mac 
```bash
mkdir /path_to_byteir/build
cd /path_to_byteir/build/

# build ByteIR
cmake ../cmake/ -G Ninja \
                -DCMAKE_BUILD_TYPE=Release \
                -DLLVM_INSTALL_PATH=path_to_LLVM_installed_or_built_directory \
                -DLLVM_EXTERNAL_LIT=lit_executatble_location # or using $(which lit), this is optional for external lit 

cmake --build . --target all
```
### Windows 
```bash
mkdir /path_to_byteir/build
cd /path_to_byteir/build/

# build ByteIR
cmake ../cmake/ -G "Visual Studio 16 2019" -A x64 \
                -DCMAKE_BUILD_TYPE=Release \
                -DLLVM_INSTALL_PATH=path_to_LLVM_installed_or_built_directory \
                -DLLVM_EXTERNAL_LIT=lit_location # this is optional for external lit 

cmake --build . --target all
```

## Testing 
This command runs all ByteIR unit tests:
```bash
cmake --build . --target check-byteir
```
ByteIR relies on ```llvm-lit``` and ```FileCheck``` for testing.
For more information, you can refer to [this page](https://www.llvm.org/docs/CommandGuide/FileCheck.html)
All the tests are placed in the folder ```byteir/test```.


## Install (Optional)
```bash
cmake --install . --prefix path_to_install_BYTEIR
```

## Pack Python Wheel (Optional)
```bash
cmake --build . --target byteir-python-pack
# byteir-*.whl in /path_to_byteir/build/python/dist/
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
