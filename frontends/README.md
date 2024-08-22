# ByteIR Frontends 

ByteIR Frontends includes Tensorflow, PyTorch, and ONNX.

Each of them can generates stablehlo dialects from the corresponding frontend.

Each frontend can be built independently with the corresponding requirement and dependencies. 
Note it may or may not be guaranteed using the same version of dependencies, e.g. LLVM, with the other frontend, due to convenience of development.

But each frontend will be guaranteed to generate compatible stablehlo format with the ByteIR compiler.

## [TensorFlow](tf-frontend/README.md)
tf graph --> tf dialect --> stablehlo dialect pipeline

## [PyTorch](torch-frontend/README.md)
PyTorch --> torch dialect --> stablehlo dialect pipeline

## [ONNX](onnx-frontend/README.md)
onnx graph --> onnx dialect --> stablehlo dialect




