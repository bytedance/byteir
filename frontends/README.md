# ByteIR Frontends 

ByteIR Frontends includes Tensorflow, PyTorch, and ONNX.

Each of them can generates mhlo dialects from the corresponding frontend.

Each frontend can be built independently with the corresponding requirement and dependencies. 
Note it may or may not be guaranteed using the same version of dependencies, e.g. LLVM, with the other frontend, due to convenience of development.

But each frontend will be guaranteed to generate compatible mhlo format with the ByteIR compiler.

## [TensorFlow](tf-frontend/README.md)
tf graph --> tf dialect --> mhlo dialect pipeline

## [PyTorch]()
TBD

## [ONNX](onnx-frontend/README.md)
onnx graph --> onnx dialect --> mhlo dialect




