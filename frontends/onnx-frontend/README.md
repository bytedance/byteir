# ONNX-Frontend

ONNX-Frontend is a project to build customized onnx graph --> onnx dialect --> mhlo dialect pipeline.

## Quick Start

### Prepare
```
git clone https://github.com/bytedance/byteir.git
cd byteir/frontends/onnx-frontend
export ONNX_FRONTEND_ROOT=$(pwd)
export ONNX_MLIR_ROOT=$ONNX_FRONTEND_ROOT/third_party/onnx-mlir
export ONNX_OFFICIAL_ROOT=$ONNX_MLIR_ROOT/third_party/onnx

git submodule update --init --recursive $ONNX_MLIR_ROOT

cd $ONNX_MLIR_ROOT && git apply $ONNX_FRONTEND_ROOT/third_party/patches/OnnxMlir*.patch
cd $ONNX_OFFICIAL_ROOT && git apply $ONNX_FRONTEND_ROOT/third_party/patches/OnnxOfficial*.patch

pip3 install lit>=14.0.0
```

### Build onnx-frontend from source code and run
First, build MLIR (llvm-project commit `9acc2f37bdfce08ca0c2faec03392db10d1bb7a9`) with cmake option `-DLLVM_ENABLE_RTTI=ON`.
Then,
```
mkdir $ONNX_FRONTEND_ROOT/build
cd $ONNX_FRONTEND_ROOT/build
cmake "-H$ONNX_FRONTEND_ROOT" \
      "-B$ONNX_FRONTEND_ROOT/build" \
      -GNinja \
      -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
      -DPython3_ROOT_DIR=/usr/bin/python3.7 \
      -DPY_VERSION=3 \
      -DMLIR_DIR="${YOUR_MLIR_DIR}/lib/cmake/mlir" \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_EXTERNAL_LIT=$(which lit)

cmake --build "$ONNX_FRONTEND_ROOT/build" --config Release --target onnx-frontend onnx-frontend-opt
```
After the above commands succeed, an `onnx-frontend` executable should appear in `$ONNX_FRONTEND_ROOT/build/onnx-frontend/src/`.

### Run lit test
```
cmake --build "$ONNX_FRONTEND_ROOT/build" --target check-of-lit
```

### Example
```
onnx-frontend model.onnx -batch-size=1 -invokeOnnxVersionConverter -o model.mhlo.mlir
```

## Contributing

### How to Add ONNX-2-MHLO conversion
- Before you start
  - Build onnx-mlir, see [this](https://github.com/onnx/onnx-mlir/blob/main/docs/BuildOnLinuxOSX.md).
  - Learn the basic knowledge of MLIR, especially the [Pattern Rewritten doc](https://mlir.llvm.org/docs/PatternRewriter/) and the [DDR doc](https://mlir.llvm.org/docs/DeclarativeRewrites/).

- Workflow
  - Develop new patterns in third_party/onnx-mlir
  - Save changes into a new .patch file under `third_party/patches` folder

- Steps to add a onnx-2-mhlo convertion pattern

  - Take a look at the [definition of the onnx op you want to convert](https://github.com/onnx/onnx-mlir/blob/main/src/Dialect/ONNX/ONNXOps.td.inc), and onnx is more like a coarse-grained op collection.
  - Take a look at the [corresponding mhlo op definition](https://github.com/tensorflow/mlir-hlo/blob/master/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.td), and mhlo is more like a fine-grained op collection.
  - Implement the pattern under `onnx-mlir/src/Conversion/` directory, note that if there's no 1:1 mapping from onnx to mhlo, converting the coarse-grained onnx op to fine-grained onnx ops is preferred before converting from onnx dialect to mhlo dialect. In this way, the fine-grained onnx ops to mhlo ops' conversion could be reused.
  - Populate the implemented pattern into convert-onnx-to-mhlo pass
  - Add lit test under `onnx-mlir/test/mlir/conversion/onnx_to_mhlo` directory

- Developing tips
  - Reference how [tensorflow convert coarse-grained ops to fine-grained ops](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.cc). 
  - Reference how [tensorflow lower tf dialect to mhlo dialect](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/mlir/xla/transforms/legalize_tf.cc).
