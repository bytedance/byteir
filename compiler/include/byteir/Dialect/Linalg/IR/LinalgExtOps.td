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

  }];
}

class LinalgExtStructuredBase_Op<string mnemonic, list<Trait> props>
  : LinalgExt_BaseOp<mnemonic, !listconcat([
       SingleBlockImplicitTerminator<"::mlir::linalg_ext::YieldOp">,
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
      resutl = exp(data - max_new)/ accum_new.
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

def LinalgExt_BatchMatmulOp : LinalgExtStructuredBase_Op<"batch_matmul", 
    [DeclareOpInterfaceMethods<TilingInterface,
        ["generateResultTileValue",
         "getIterationDomain",
         "getLoopIteratorTypes",
         "getResultTilePosition",
         "getTiledImplementation"]>]> {
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

    // Additional functions
    int64_t getFullRank() {
      return getInit().getType().cast<ShapedType>().getRank() + 1;
    }
    
  }];
}

#endif // BYTEIR_DIALECT_LINALG_IR_LINALG_EXT_OPS