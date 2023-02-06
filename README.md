# The ByteIR Project

The ByteIR Project is a ByteDance model compilation solution.
ByteIR includes compiler, runtime, and frontends, and provides an end-to-end model compilation solution.

Although all ByteIR components (compiler/runtime/frontends) are together to provide an end-to-end solution, and 
all under the same umbrella of this repository, 
each component technically can perform independently.

## Project Status
ByteIR is still in its early phase. 
In this phase, we are aiming to provide well-defined, necessary building blocks and infrastructure support for model compilation in a wide-ranage of deep learning accelerators as well as general-purpose CPUs and GPUs.
Therefore, highly-tuned kernels for specific achiecture might not have been prioritized. 
For sure, any feedback for prioritizing specific achiecture or correpsonding contribution are wellcome.

## [Compiler](compiler/README.md)

ByteIR Compiler is an MLIR-based compiler for CPU/GPU/ASIC.

## [Runtime](runtime/README.md)

ByteIR Runtime is a common, lightweight runtime, capable to serving both existing kernels and ByteIR compiler generated kernels.

## [Frontends](frontends/README.md)

ByteIR Frontends includes Tensorflow, PyTorch, and ONNX.


## Components Communication Interface
Each ByteIR component technically can perform independently.
There are pre-defined communication interface between components.

### MHLO between frontends and compiler
ByteIR frontends and ByteIR compiler communicates through mhlo dialect, which version might be updated during development.

This also implies whatever frontend generating mhlo with a compatible verison can work with ByteIR compiler, and also whatever compiler consuming mhlo with a compatible verison can work with ByteIR frontends.

### ByRE between compiler and runtime

ByteIR compiler and ByteIR runtime communicates through ByRE format, which version might be updated during development.
ByRE dialect is defined as a kind of ByRE format in ByteIR compiler, 
currently supporting a textual form for ByteIR compiler and runtime.

The bytecode form of ByRE dialect with versioning will come very soon.
Other ByRE formats are under development.


## [License](LICENSE)

The ByteIR Project is under the Apache License v2.0
