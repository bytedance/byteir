# ONNX-Frontend

ONNX-Frontend is a project to build customized onnx graph --> onnx dialect --> stablehlo dialect pipeline.

## Quick Start

### Prerequisites
```
python >= 3.7
gcc >= 6.4
protobuf >= 4.21.12
cmake >= 3.13.4
make >= 4.2.1 or ninja >= 1.10.2
java >= 1.11 (optional)
```

Look [here](https://github.com/onnx/onnx-mlir/blob/main/docs/Prerequisite.md) for help to set up the prerequisite software.


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

cd $ONNX_FRONTEND_ROOT
python3 -m pip install -r requirements.txt
```

### Build onnx-frontend from source code and run
Firstly, build MLIR (llvm-project commit `b2cdf3cc4c08729d0ff582d55e40793a20bbcdcc`) with cmake option `-DLLVM_ENABLE_RTTI=ON`.
```
git clone -n https://github.com/llvm/llvm-project.git
cd llvm-project && git checkout b2cdf3cc4c08729d0ff582d55e40793a20bbcdcc && cd ..
```

```
mkdir llvm-project/build
cd llvm-project/build

cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_TARGETS_TO_BUILD="host" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_ENABLE_RTTI=ON \
   -DLLVM_ENABLE_LIBEDIT=OFF

cmake --build . -- ${MAKEFLAGS}
cmake --build . --target check-mlir

cd ../..
MLIR_DIR=$(pwd)/llvm-project/build/lib/cmake/mlir
```

Then,
```
mkdir $ONNX_FRONTEND_ROOT/build
cd $ONNX_FRONTEND_ROOT/build
cmake "-H$ONNX_FRONTEND_ROOT" \
      "-B$ONNX_FRONTEND_ROOT/build" \
      -GNinja \
      -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
      -DPython3_ROOT_DIR=$(which python3) \
      -DPY_VERSION=3 \
      -DMLIR_DIR=${MLIR_DIR} \
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
onnx-frontend model.onnx -batch-size=1 -invokeOnnxVersionConverter -o model.stablehlo.mlir
```

## Contributing

### How to Add ONNX-2-STABLEHLO conversion
- Before you start
  - Build onnx-mlir, see [this](https://github.com/onnx/onnx-mlir/blob/main/docs/BuildOnLinuxOSX.md).
  - Learn the basic knowledge of MLIR, especially the [Pattern Rewritten doc](https://mlir.llvm.org/docs/PatternRewriter/) and the [DDR doc](https://mlir.llvm.org/docs/DeclarativeRewrites/).

- Workflow
  - Develop new patterns in third_party/onnx-mlir
  - Save changes into a new .patch file under `third_party/patches` folder

- Steps to add a onnx-2-stablehlo convertion pattern

  - Take a look at the [definition of the onnx op you want to convert](https://github.com/onnx/onnx-mlir/blob/main/src/Dialect/ONNX/ONNXOps.td.inc), and onnx is more like a coarse-grained op collection.
  - Take a look at the [corresponding stablehlo op definition](https://github.com/openxla/stablehlo/blob/main/stablehlo/dialect/StablehloOps.td), and stablehlo is more like a fine-grained op collection.
  - Implement the pattern under `onnx-mlir/src/Conversion/` directory, note that if there's no 1:1 mapping from onnx to stablehlo, converting the coarse-grained onnx op to fine-grained onnx ops is preferred before converting from onnx dialect to stablehlo dialect. In this way, the fine-grained onnx ops to stablehlo ops' conversion could be reused.
  - Populate the implemented pattern into convert-onnx-to-stablehlo pass
  - Add lit test under `onnx-mlir/test/mlir/conversion/onnx_to_stablehlo` directory

- Developing tips
  - Reference how [tensorflow convert coarse-grained ops to fine-grained ops](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.cc).
  - Reference how [tensorflow lower tf dialect to mhlo dialect](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/mlir/xla/transforms/legalize_tf.cc).
