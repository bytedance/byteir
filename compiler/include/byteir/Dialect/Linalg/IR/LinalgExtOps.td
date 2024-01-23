//===-- LinalgExtOps.td ---------------------------------------------------===//
//
// Copyright 2022 ByteDance Ltd. and/or its affiliates. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
// Some code comes from LinalgExtOps.td in IREE project
// Original license:
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Some code comes from LinalgStructuredOps.td in LLVM project
// Original license:
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_LINALG_IR_LINALG_EXT_OPS
#define BYTEIR_DIALECT_LINALG_IR_LINALG_EXT_OPS

include "byteir/Dialect/Linalg/IR/LinalgExtBase.td"
include "byteir/Dialect/Linalg/IR/LinalgExtInterfaces.td"
include "mlir/Dialect/Linalg/IR/LinalgInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/TilingInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"

// Base class for the operation in this dialect
class LinalgExt_BaseOp<string mnemonic, list<Trait> traits = []> :
    Op<LinalgExt_Dialect, mnemonic, traits> {
}

class LinalgExt_Op<string mnemonic, list<Trait> traits = []> :
    LinalgExt_BaseOp<mnemonic, !listconcat(traits,
        [AttrSizedOperandSegments,
         DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
         DestinationStyleOpInterface,
         LinalgExtInterface,
  ])> {

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
  code extraLinalgExtOpClassDeclaration = [{
    // Method to implement for specifying output range for
    // DestinationStyleOpInterface
    std::pair<int64_t, int64_t> getDpsInitsPositionRange() {
      std::pair<unsigned, unsigned> outputsIndexAndLength =
        getODSOperandIndexAndLength(1);
      return std::make_pair<int64_t, int64_t>(
          outputsIndexAndLength.first,
          outputsIndexAndLength.first + outputsIndexAndLength.second);
    }

    MutableOperandRange getDpsInitsMutable() { return getOutputsMutable(); }
  }];
}

class LinalgExtStructuredBase_Op<string mnemonic, list<Trait> props>
  : LinalgExt_BaseOp<mnemonic, !listconcat([
       SingleBlockImplicitTerminator<"::mlir::linalg::YieldOp">,
       DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
       DestinationStyleOpInterface,
       LinalgStructuredInterface,
       ReifyRankedShapedTypeOpInterface], props)> {
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
  code extraLinalgExtOpClassDeclaration = [{
    // Currently all the ops in linalg_ext has no region builder
    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return nullptr;
    }

    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() {
      return !this->getBody()->getOps<linalg::IndexOp>().empty();
    }

    // Declare functions necessary for LinalgStructuredInterface.
    ArrayAttr getIndexingMaps();
    SmallVector<utils::IteratorType> getIteratorTypesArray();
    std::string getLibraryCallName() {
      return "op_has_no_registered_library_name";
    }

    // Define functions for ReifyRankedShapedTypeOpInterface
     LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return llvm::cast<linalg::LinalgOp>(getOperation()).reifyResultShapes(b,
          reifiedReturnShapes);
    }

  }];
}

//===----------------------------------------------------------------------===//
// Basic ops
//===----------------------------------------------------------------------===//

def LinalgExt_YieldOp : LinalgExt_BaseOp<"yield", [Pure, ReturnLike, Terminator]> {
  let summary = "LinalgExt yield op";
  let description = [{
    `linalg_ext.yield` is a special terminator operation for blocks inside
    regions in `linalg_ext` ops.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins), [{ /* nothing to do */ }]>,
  ];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def LinalgExt_AliasOp : LinalgExt_BaseOp<"alias", [Pure]> {
  let summary = "LinalgExt alias op";
  let description = [{
    `linalg_ext.alias` is a special op  to represent alias in basic block
    It will be resolved in a special rewrite pattern. 
    It won't be removed in the regular canonicalizer.
  }];

  let arguments = (ins AnyType:$operand);
  let results = (outs AnyType:$result);

  let builders = [
    OpBuilder<(ins "Value":$operand), [{
      build($_builder, $_state, operand.getType(), operand);
    }]> 
  ];

  let hasVerifier = 1;

  let assemblyFormat = " `(` $operand `:` type($operand) `)` attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Advanced ops
//===----------------------------------------------------------------------===//

def LinalgExt_SoftmaxOp : LinalgExt_Op<"softmax",
    [DeclareOpInterfaceMethods<LinalgExtInterface, 
        ["isValidTiling",
         "isValidTiledProducerOp",
         "makeValidTiledConsumerOps",
         "isResultLoopInvariant",
         "isOperandRead"]>,
     DeclareOpInterfaceMethods<TilingInterface,
        ["generateResultTileValue",
         "getIterationDomain",
         "getLoopIteratorTypes",
         "getResultTilePosition", 
         "getTiledImplementation"]>]> {

  let summary = "Softmax Op";
  let description = [{
    Computes softmax along a given dimension.
    Tensor:
      [result, max, accum, scale] = softmax {dimension} ins(data) outs(result, max, accum, scale);
    Memeref
      softmax {dimension} ins(data) outs(result, scale, max, accum)
    Computation: 
      max_new = max(max_old, max(data along dimension)).
      accum_new = accum_old * exp(max_old - max_new) + sum(exp(data - max_new)). 
      scale = accum_old * exp(max_old - max_new) / accum_new.
      result = exp(data - max_new)/ accum_new.
  }];

  let arguments = (ins
      Variadic<AnyRankedTensorOrMemRefType>:$inputs,
      Variadic<AnyRankedTensorOrMemRefType>:$outputs,
      I64Attr:$dimension
  );

  let results = (outs Variadic<AnyRankedTensor>:$results);

  let assemblyFormat = [{
    attr-dict 
    `dimension` `(` $dimension `)`
    (`ins` `(` $inputs^ `:` type($inputs) `)`)?
    (`outs` `(` $outputs^ `:` type($outputs) `)`)?
    (`:` type($results)^)?
  }];
  let hasFolder = 1;

  let extraClassDeclaration = extraLinalgExtOpClassDeclaration # [{
    Value input() {
      return getInputOperand(0)->get();
    }
    Value output() {
      return getOutputOperand(0)->get();
    }
    Value max() {
      return getOutputOperand(1)->get();
    }
    Value accumulator() {
      return getOutputOperand(2)->get();
    }
    Value scale() {
      return getOutputOperand(3)->get();
    }
    ShapedType getOperandType() {
      return output().getType().cast<ShapedType>();
    }
    int64_t getOperandRank() {
      return getOperandType().getRank();
    }
  }];

}

def LinalgExt_UnnormalizedSoftmaxOp : LinalgExt_Op<"unnorm_softmax",
    [DeclareOpInterfaceMethods<LinalgExtInterface, 
        ["isValidTiling",
         "isValidTiledProducerOp",
         "makeValidTiledConsumerOps",
         "isResultLoopInvariant",
         "isOperandRead"]>,
     DeclareOpInterfaceMethods<TilingInterface,
        ["generateResultTileValue",
         "getIterationDomain",
         "getLoopIteratorTypes",
         "getResultTilePosition", 
         "getTiledImplementation"]>]> {

  let summary = "Unnormalized Softmax Op";
  let description = [{
    Computes unnormalized softmax (unnorm_softmax) along a given dimension.
    Tensor:
      [result, max, accum, scale] = unnorm_softmax {dimension} ins(data) outs(result, max, accum, scale);
    Memeref
      unnorm_softmax {dimension} ins(data) outs(result, scale, max, accum)
    Computation: 
      max_new = max(max_old, max(data along dimension)).
      accum_new = accum_old * exp(max_old - max_new) + sum(exp(data - max_new)). 
      scale = exp(max_old - max_new).
      result = exp(data - max_new).
  }];

  let arguments = (ins
      Variadic<AnyRankedTensorOrMemRefType>:$inputs,
      Variadic<AnyRankedTensorOrMemRefType>:$outputs,
      I64Attr:$dimension
  );

  let results = (outs Variadic<AnyRankedTensor>:$results);

  let assemblyFormat = [{
    attr-dict 
    `dimension` `(` $dimension `)`
    (`ins` `(` $inputs^ `:` type($inputs) `)`)?
    (`outs` `(` $outputs^ `:` type($outputs) `)`)?
    (`:` type($results)^)?
  }];
  let hasFolder = 1;

  let extraClassDeclaration = extraLinalgExtOpClassDeclaration # [{
    Value input() {
      return getInputOperand(0)->get();
    }
    Value output() {
      return getOutputOperand(0)->get();
    }
    Value max() {
      return getOutputOperand(1)->get();
    }
    Value accumulator() {
      return getOutputOperand(2)->get();
    }
    Value scale() {
      return getOutputOperand(3)->get();
    }
    ShapedType getOperandType() {
      return output().getType().cast<ShapedType>();
    }
    int64_t getOperandRank() {
      return getOperandType().getRank();
    }
  }];

}

def LinalgExt_DiagOp : LinalgExt_Op<"diag", 
    [DeclareOpInterfaceMethods<LinalgExtInterface>]> {

  let summary = "Diag Op";
  let description = [{
    diag(x) presents a diag matrix from a vector.
    it only works on the last dimension.
    it is an intermediate op for softmax's fusion

    1D Example:
    ```mlir
    %1 = linalg_ext.diag 
    ins(%arg0 : tensor<1024xf32>) outs(%0 : tensor<1024x1024xf32>) : tensor<1024x1024xf32>
    ```

    2D Example
    ```mlir
    %1 = linalg_ext.diag
    ins(%arg0 : tensor<512x1024xf32>) outs(%0 : tensor<512x1024x1024xf32>) : tensor<512x1024x1024xf32>
    ```
  }];

  let arguments = (ins
      Variadic<AnyRankedTensorOrMemRefType>:$inputs,
      Variadic<AnyRankedTensorOrMemRefType>:$outputs
  );

  let results = (outs Variadic<AnyRankedTensor>:$results);

  let assemblyFormat = [{
    attr-dict 
    (`ins` `(` $inputs^ `:` type($inputs) `)`)?
    (`outs` `(` $outputs^ `:` type($outputs) `)`)?
    (`:` type($results)^)?
  }];
  let hasFolder = 1;

  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$output), [{
      build($_builder, $_state, {output.getType()}, {input}, {output});
    }]> 
  ];

  let extraClassDeclaration = extraLinalgExtOpClassDeclaration # [{
    Value input() {
      return getInputOperand(0)->get();
    }
    Value output() {
      return getOutputOperand(0)->get();
    }
    ShapedType getOperandType() {
      return output().getType().cast<ShapedType>();
    }
    int64_t getOperandRank() {
      return getOperandType().getRank();
    }

    static Type getDiagType(ShapedType type);
  }];

}

def LinalgExt_ScanOp : LinalgExt_Op<"scan",
    [DeclareOpInterfaceMethods<LinalgExtInterface,
      ["isValidTiledProducerOp",
       "isResultLoopInvariant",
       "isOperandRead"]>,
     DeclareOpInterfaceMethods<TilingInterface,
      ["generateResultTileValue",
       "getIterationDomain",
       "getLoopIteratorTypes",
       "getResultTilePosition",
       "getTiledImplementation"]>]> {
  let summary = "Scan operator";
  let description = [{
    Computes the inclusive/exclusive scan along a given dimension.
  }];

  let arguments = (ins Variadic<AnyShaped>:$inputs,
                       Variadic<AnyShaped>:$outputs,
                       I64Attr:$dimension,
                       BoolAttr:$inclusive
  );

  let builders = [
    OpBuilder<(ins "ValueRange":$inputs, "ValueRange":$outputs,
      CArg<"int64_t", "0">:$dimension, CArg<"bool", "true">:$inclusive)>
  ];

  let results = (outs Variadic<AnyRankedTensor>:$results);
  let regions = (region AnyRegion:$region);
  let hasFolder = 1;
  let assemblyFormat = [{
    attr-dict
    `dimension` `(` $dimension `)`
    `inclusive` `(` $inclusive `)`
    `ins` `(` $inputs `:` type($inputs) `)`
    `outs` `(` $outputs `:` type($outputs) `)`
    $region (`->` type($results)^)?
  }];

  let extraClassDeclaration = extraLinalgExtOpClassDeclaration # [{
    Value input() {
      return getInputOperand(0)->get();
    }
    Value accumulator() {
      return getOutputOperand(1)->get();
    }
    Value output() {
      return getOutputOperand(0)->get();
    }
    ShapedType getOperandType() {
      return input().getType().cast<ShapedType>();
    }
    int64_t getOperandRank() {
      return getOperandType().getRank();
    }
  }];
}

def LinalgExt_ScatterOp : LinalgExt_Op<"scatter",
    [DeclareOpInterfaceMethods<LinalgExtInterface,
      ["isValidTiledProducerOp",
       "isResultLoopInvariant",
       "isOperandRead"]>,
     DeclareOpInterfaceMethods<TilingInterface,
      ["generateResultTileValue",
       "getIterationDomain",
       "getLoopIteratorTypes",
       "getResultTilePosition",
       "getTiledImplementation",
       "generateScalarImplementation"]>]> {
  let summary = "scatter operator";
  let description = [{
    Foreach scatter index vector I iterated over frist `rank(indices) - 1`
    dimensions along `indices`, the `batch_indexes` was defined in:

    batch_indexes = (b[0], b[1], ..., b[rank(indices) - 2]) and it is a
    `rank(indices) - 1`-length vector

    `src` would be updated with `update` as follow:

      let I = indices[batch_indexes] is a 1D vector
      let U = update[batch_indexes] is a `rank(update) - rank(indices) + 1`D tensor
      let S = src[I] is a `rank(src) - rank(indices) + 1`D tensor
      let `compute` be the computation function defined in op region
      S = compute(S, U)

    therefore it meets the following requirements:

      1. `indices`, `update` and `src` are all ranked tensors or memrefs
      2. the first `rank(indices) - 1` dimensions of `indices` and `update`
         are compatible
      3. the last `rank(update) - rank(indices) + 1` dimensions of `update` and `src`
         are compatible
      4. the last dimension of the indices, denoted as dim(indices, rank(indices) - 1),
         should be static and the rank of `src` is equal to
         `dim(indices, rank(indices) - 1) + rank(update) - rank(indices) + 1`


    For example:

      linalg_ext.scatter
        ins(%indices, %update: tensor<100x2xi64>, tensor<100x32x64xf32>)
        outs(%src: tensor<2x3x32x64xf32>) {
          ^bb0(%arg0: f32, %arg1: f32):
            %0 = arith.addf %arg0, %arg1 : f32
            linalg_ext.yield %0 : f32
        } -> tensor<2x3x32x64xf32>
  }];
  let arguments = (ins
    Variadic<AnyRankedTensorOrMemRefType>:$inputs,
    Variadic<AnyRankedTensorOrMemRefType>:$outputs
    // TODO: support unique indices
  );

  let results = (outs Variadic<AnyRankedTensor>:$results);

  let regions = (region SizedRegion<1>:$region);
  let assemblyFormat = [{
    attr-dict
    `ins` `(` $inputs `:` type($inputs) `)`
    `outs` `(` $outputs `:` type($outputs) `)`
    $region (`->` type($results)^)?
  }];

  let extraClassDeclaration = extraLinalgExtOpClassDeclaration # [{
    Value indices() {
      return getInputOperand(0)->get();
    }
    Value update() {
      return getInputOperand(1)->get();
    }
    Value src() {
      return getOutputOperand(0)->get();
    }
    ShapedType getUpdateType() {
      return update().getType().cast<ShapedType>();
    }
    ShapedType getIndicesType() {
      return indices().getType().cast<ShapedType>();
    }
    ShapedType getSrcType() {
      return src().getType().cast<ShapedType>();
    }
    int64_t getUpdateRank() {
      return getUpdateType().getRank();
    }
    int64_t getIndicesRank() {
      return getIndicesType().getRank();
    }
    int64_t getSrcRank() {
      return getSrcType().getRank();
    }
  }];
}

def LinalgExt_TopkOp : LinalgExt_Op<"topk",
    [DeclareOpInterfaceMethods<LinalgExtInterface,
      ["isValidTiledProducerOp",
       "isResultLoopInvariant",
       "isOperandRead"]>,
     DeclareOpInterfaceMethods<TilingInterface,
      ["generateResultTileValue",
       "getIterationDomain",
       "getLoopIteratorTypes",
       "getResultTilePosition",
       "getTiledImplementation"]>]> {
  let summary = "topk operator";
  let description = [{
    Computes the topk along a given dimension.
  }];

  let arguments = (ins Variadic<AnyShaped>:$inputs,
                       Variadic<AnyShaped>:$outputs,
                       I64Attr:$dimension
  );

  let builders = [
    OpBuilder<(ins "ValueRange":$inputs, "ValueRange":$outputs,
      CArg<"int64_t", "0">:$dimension)>
  ];

  let results = (outs Variadic<AnyRankedTensor>:$results);
  let regions = (region AnyRegion:$region);
  let hasFolder = 1;
  let assemblyFormat = [{
    attr-dict
    `dimension` `(` $dimension `)`
    `ins` `(` $inputs `:` type($inputs) `)`
    `outs` `(` $outputs `:` type($outputs) `)`
    $region (`->` type($results)^)?
  }];

  let extraClassDeclaration = extraLinalgExtOpClassDeclaration # [{
    Value values() {
      return getInputOperand(0)->get();
    }
    std::optional<Value> indices() {
      if (getNumInputs() < 2) {
        return {};
      } else {
        return getInputOperand(1)->get();
      }
    }
    Value outputValues() {
      return getOutputOperand(0)->get();
    }
    Value outputIndices() {
      return getOutputOperand(1)->get();
    }
    ShapedType getInputType() {
      return values().getType().cast<ShapedType>();
    }
    int64_t getInputRank() {
      return getInputType().getRank();
    }
  }];
}

def LinalgExt_CustomOp : LinalgExt_Op<"custom",
    [DeclareOpInterfaceMethods<LinalgExtInterface>,
     DeclareOpInterfaceMethods<TilingInterface,
        ["generateResultTileValue",
         "getIterationDomain",
         "getLoopIteratorTypes",
         "getResultTilePosition", 
         "getTiledImplementation"]>,
      SingleBlockImplicitTerminator<"::mlir::linalg_ext::YieldOp">,
    ]> {

  let summary = "custom op wrapper";
  let description = [{
    Custom op wrapper. TODO add an exmaple here.
  }];

  let arguments = (ins
      Variadic<AnyRankedTensorOrMemRefType>:$inputs,
      Variadic<AnyRankedTensorOrMemRefType>:$outputs
  );
  let results = (outs Variadic<AnyRankedTensor>:$results);
   let regions = (region AnyRegion:$region);

  let assemblyFormat = [{
    attr-dict 
    (`ins` `(` $inputs^ `:` type($inputs) `)`)?
    `outs` `(` $outputs `:` type($outputs) `)`
    $region (`->` type($results)^)?
  }];

  let extraClassDeclaration = extraLinalgExtOpClassDeclaration # [{
  }];
}

def LinalgExt_LayerNormOp : LinalgExt_Op<"layer_norm", 
    [DeclareOpInterfaceMethods<LinalgExtInterface, 
        ["isValidTiling"]>,
     DeclareOpInterfaceMethods<TilingInterface,
        ["generateResultTileValue",
         "getIterationDomain",
         "getLoopIteratorTypes",
         "getResultTilePosition",
         "getTiledImplementation"]>]> {
  let summary = "Layer normalization";
  let description = [{
    Applies Layer Normalization over a mini-batch of inputs.

    - Operands:
    - input: Tensor
    - weight: Optional<Tensor>
    - bias: Optional<Tensor>

    - Results(1 or 3):
    - output: Tensor 
    - mean: Optional<Tensor>
    - inv_std_dev: Optional<Tensor>

    output = (x - E[x])/sqrt(Var[x] + epsilon) * weight + bias
  }];

  let arguments = (ins
      Variadic<AnyRankedTensorOrMemRefType>:$inputs,
      Variadic<AnyRankedTensorOrMemRefType>:$outputs,
      I64ArrayAttr:$axis,
      F64Attr:$epsilon
  );

  let results = (outs Variadic<AnyRankedTensor>:$results);

  let assemblyFormat = [{
    attr-dict 
    `axis` `(` $axis `)`
    `epsilon` `(` $epsilon `)`
    (`ins` `(` $inputs^ `:` type($inputs) `)`)?
    (`outs` `(` $outputs^ `:` type($outputs) `)`)?
    (`:` type($results)^)?
  }];

  let extraClassDeclaration = extraLinalgExtOpClassDeclaration # [{
    Value input() {
      return getInputOperand(0)->get();
    }
    Value weight() {
      return getInputOperand(1)->get();
    }
    Value bias() {
      return getInputOperand(2)->get();
    }
    Value output() {
      return getOutputOperand(0)->get();
    }
    Value mean() {
      return getOutputOperand(1)->get();
    }
    Value rstd() {
      return getOutputOperand(2)->get();
    }
    ShapedType getOperandType(int64_t idx) {
      return getInputOperand(idx)->get().getType().cast<ShapedType>();
    }
    int64_t getOperandRank(int64_t idx) {
      return getOperandType(idx).getRank();
    }
    SmallVector<int64_t> getIntAxis(){
      auto axises = llvm::to_vector(llvm::map_range(
        getAxis().getAsRange<IntegerAttr>(),
        [&](IntegerAttr intAttr) { return intAttr.getInt(); }));
      return axises;
    }
  }];
}

def LinalgExt_BatchMatmulOp : LinalgExtStructuredBase_Op<"batch_matmul", 
    [DeclareOpInterfaceMethods<TilingInterface,
        ["generateResultTileValue",
         "getIterationDomain",
         "getLoopIteratorTypes",
         "getResultTilePosition",
         "getTiledImplementation"]>,
     DeclareOpInterfaceMethods<PartialReductionOpInterface,
     ["generateInitialTensorForPartialReduction",
      "tileToPartialReduction",
      "mergeReductions"]>]> {
  let summary = "Batched matrix multiplication";
  let description = [{
    Performs a batched matrix multiplication of two N-D (N >= 3) inputs.

    Numeric casting is performed on the operands to the inner multiply, promoting
    them to the same data type as the accumulator/output.
  }];

  let arguments = (ins AnyRankedTensorOrMemRefType:$lhs,
                       AnyRankedTensorOrMemRefType:$rhs,
                       AnyRankedTensorOrMemRefType:$init,
                       StrAttr:$layout
  );
  let results = (outs
    Variadic<AnyRankedTensor>:$results
  );
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs, "Value":$init,
        "StringAttr":$layout, CArg<"ArrayRef<NamedAttribute>",
        "{}">:$attributes)>,
    OpBuilder<(ins "Value":$lhs, "Value":$rhs, "Value":$init,
        "StringRef":$layout, CArg<"ArrayRef<NamedAttribute>",
        "{}">:$attributes)>,
  ];

  let extraClassDeclaration = extraLinalgExtOpClassDeclaration # [{
    // Implement functions necessary for DestinationStyleOpInterface.
    std::pair<int64_t, int64_t> getDpsInitsPositionRange() {
      int64_t getNumOperands = this->getNumOperands();
      return {getNumOperands - 1, getNumOperands};
    }

    MutableOperandRange getDpsInitsMutable() { return getInitMutable(); }

    // Additional functions
    int64_t getFullRank() {
      return getInit().getType().cast<ShapedType>().getRank() + 1;
    }
    
  }];
}

#endif // BYTEIR_DIALECT_LINALG_IR_LINALG_EXT_OPS