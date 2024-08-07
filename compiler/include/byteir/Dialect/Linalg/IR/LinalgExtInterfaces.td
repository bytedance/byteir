//===-- LinalgExtInterfaces.td --------------------------------------------===//
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
// Some code comes from LinalgExtInterfaces.td in IREE project
// Original license:
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_LINALG_IR_LINALG_EXT_INTERFACES
#define BYTEIR_DIALECT_LINALG_IR_LINALG_EXT_INTERFACES

include "byteir/Dialect/Linalg/IR/LinalgExtBase.td"

// The interface is a subset of LinalgStructuredInterface.
def LinalgExtInterface : OpInterface<"LinalgExtOp"> {
  let methods = [
    //===------------------------------------------------------------------===//
    // Num input/output arguments handling.
    //===------------------------------------------------------------------===//
    // `inputs` must be defined by each op that wants to implement the
    // LinalgStructuredInterface.
    InterfaceMethod<
      /*desc=*/[{
        Return the input shape operands.
      }],
      /*retTy=*/"ValueRange",
      /*methodName=*/"getInputs",
      /*args=*/(ins)
    >,
    // These special methods rely on `inputs` and `outputs` being defined by
    // each op that wants to implement the LinalgStructuredInterface.
    InterfaceMethod<
      /*desc=*/[{
        Return the number of inputs.
      }],
      /*retTy=*/"int64_t",
      /*methodName=*/"getNumInputs",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getInputs().size();
      }]
    >,
    // `outputs` must be defined by each op that wants to implement the
    // LinalgStructuredInterface.
    InterfaceMethod<
      /*desc=*/[{
        Return the output shape operands.
      }],
      /*retTy=*/"ValueRange",
      /*methodName=*/"getOutputs",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the number of outputs.
      }],
      /*retTy=*/"int64_t",
      /*methodName=*/"getNumOutputs",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getOutputs().size();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the number of inputs and outputs.
      }],
      /*retTy=*/"int64_t",
      /*methodName=*/"getNumInputsAndOutputs",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return getNumInputs() + getNumOutputs();
      }]
    >,
    //===------------------------------------------------------------------===//
    // Input operands handling.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      /*desc=*/[{
        Return the input operands.
      }],
      /*retTy=*/"SmallVector<OpOperand *>",
      /*methodName=*/"getInputOperands",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        int64_t numInputs = getNumInputs();
        SmallVector<OpOperand *> result;
        result.reserve(numInputs);
        llvm::transform(
          this->getOperation()->getOpOperands().take_front(numInputs),
          std::back_inserter(result),
          [](OpOperand &opOperand) { return &opOperand; });
        return result;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the `i`-th input operand.
      }],
      /*retTy=*/"OpOperand*",
      /*methodName=*/"getInputOperand",
      /*args=*/(ins "int64_t":$i),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(i >= 0 && i < getNumInputs());
        return &this->getOperation()->getOpOperand(i);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the subset of input operands that are of buffer type.
      }],
      /*retTy=*/"SmallVector<OpOperand *>",
      /*methodName=*/"getInputBufferOperands",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        SmallVector<OpOperand *> result;
        result.reserve(getNumInputs());
        llvm::copy_if(getInputOperands(),
          std::back_inserter(result),
          [](OpOperand *opOperand) {
            return isa<MemRefType>(opOperand->get().getType());
          });
        return result;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the subset of input operands that are of tensor type.
      }],
      /*retTy=*/"SmallVector<OpOperand *>",
      /*methodName=*/"getInputTensorOperands",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        SmallVector<OpOperand *> result;
        result.reserve(getNumInputs());
        llvm::copy_if(getInputOperands(),
          std::back_inserter(result),
          [](OpOperand *opOperand) {
            return isa<RankedTensorType>(opOperand->get().getType());
          });
        return result;
      }]
    >,
    //===------------------------------------------------------------------===//
    // Output operands handling.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      /*desc=*/[{
        Return the output operands.
      }],
      /*retTy=*/"SmallVector<OpOperand *>",
      /*methodName=*/"getOutputOperands",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        int64_t numOutputs = getNumOutputs();
        SmallVector<OpOperand *> result;
        result.reserve(numOutputs);
        llvm::transform(
          this->getOperation()->getOpOperands()
            .drop_front(getNumInputs())
            .take_front(numOutputs),
          std::back_inserter(result),
          [](OpOperand &opOperand) { return &opOperand; });
        return result;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the `i`-th output operand.
      }],
      /*retTy=*/"OpOperand*",
      /*methodName=*/"getOutputOperand",
      /*args=*/(ins "int64_t":$i),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(i >= 0 && i < getNumOutputs());
        return &this->getOperation()->getOpOperand(getNumInputs() + i);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Set the `i`-th output operand.
      }],
      /*retTy=*/"void",
      /*methodName=*/"setOutputOperand",
      /*args=*/(ins "int64_t":$i, "Value":$value),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(i >= 0 && i < getNumOutputs());
        this->getOperation()->setOperand(getNumInputs() + i, value);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the subset of output operands that are of buffer type.
      }],
      /*retTy=*/"SmallVector<OpOperand *>",
      /*methodName=*/"getOutputBufferOperands",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        SmallVector<OpOperand *> result;
        result.reserve(getNumOutputs());
        llvm::copy_if(getOutputOperands(),
          std::back_inserter(result),
          [](OpOperand *opOperand) {
            return isa<MemRefType>(opOperand->get().getType());
          });
        return result;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the subset of output operands that are of tensor type.
      }],
      /*retTy=*/"SmallVector<OpOperand *>",
      /*methodName=*/"getOutputTensorOperands",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        SmallVector<OpOperand *> result;
        result.reserve(getNumOutputs());
        llvm::copy_if(getOutputOperands(),
          std::back_inserter(result),
          [](OpOperand *opOperand) {
            return isa<RankedTensorType>(opOperand->get().getType());
          });
        return result;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the types of the subset of output operands that are of buffer type.
      }],
      /*retTy=*/"SmallVector<MemRefType>",
      /*methodName=*/"getOutputBufferTypes",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        SmallVector<MemRefType> result;
        result.reserve(getNumOutputs());
        llvm::transform(getOutputBufferOperands(),
          std::back_inserter(result),
          [](OpOperand *opOperands) {
            return cast<MemRefType>(opOperands->get().getType());
          });
        return result;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the types of the subset of output operands that are of tensor type.
      }],
      /*retTy=*/"SmallVector<RankedTensorType>",
      /*methodName=*/"getOutputTensorTypes",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        SmallVector<RankedTensorType> result;
        result.reserve(getNumOutputs());
        llvm::transform(getOutputTensorOperands(),
          std::back_inserter(result),
          [](OpOperand *opOperands) {
            return cast<RankedTensorType>(opOperands->get().getType());
          });
        return result;
      }]
    >,
    //===------------------------------------------------------------------===//
    // Input and Output arguments handling.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      /*desc=*/[{
        Return the range over input and output operands.
      }],
      /*retTy=*/"SmallVector<OpOperand *>",
      /*methodName=*/"getInputAndOutputOperands",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        int64_t numInputsAndOutputs = getNumInputsAndOutputs();
        SmallVector<OpOperand *> result;
        result.reserve(numInputsAndOutputs);
        llvm::transform(
          this->getOperation()->getOpOperands()
            .take_front(numInputsAndOutputs),
          std::back_inserter(result),
          [](OpOperand &opOperand) { return &opOperand; });
        return result;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return true if `opOperand` is an input tensor.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isInputTensor",
      /*args=*/(ins "OpOperand *":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        if (!isa<RankedTensorType>(opOperand->get().getType()))
          return false;
        if (opOperand->getOperandNumber() < $_op.getNumInputs())
          return true;
        return false;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return true if `opOperand` is an output tensor.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isOutputTensor",
      /*args=*/(ins "OpOperand *":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        if (!isa<RankedTensorType>(opOperand->get().getType()))
          return false;
        if (opOperand->getOperandNumber() >= $_op.getNumInputs())
          return true;
        return false;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the `opOperand` rank or zero for scalars.
      }],
      /*retTy=*/"int64_t",
      /*methodName=*/"getRank",
      /*args=*/(ins "OpOperand*":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(opOperand->getOwner() == this->getOperation());
        if (auto shapedType =
              dyn_cast<ShapedType>(opOperand->get().getType()))
          return shapedType.getRank();
        return 0;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the `opOperand` shape or an empty vector for scalars.
      }],
      /*retTy=*/"ArrayRef<int64_t>",
      /*methodName=*/"getShape",
      /*args=*/(ins "OpOperand*":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(opOperand->getOwner() == this->getOperation());
        if (auto shapedType =
              dyn_cast<ShapedType>(opOperand->get().getType()))
          return shapedType.getShape();
        return {};
      }]
    >,
      InterfaceMethod<
      /*desc=*/[{
        Return whether operand is a read
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isOperandRead",
      /*args=*/(ins "unsigned":$number),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return number < getNumInputs();
      }]
    >,
    //===------------------------------------------------------------------===//
    // Non input and output operands handling
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      /*desc=*/[{
        Return operands that are neither inputs nor outputs.
      }],
      /*retTy=*/"SmallVector<OpOperand *>",
      /*methodName=*/"getNonInputOrOutputOperands",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        int64_t numInputsAndOutputs = getNumInputsAndOutputs();
        int64_t numOperands = this->getOperation()->getNumOperands();
        assert(numInputsAndOutputs <= numOperands);
        if (numInputsAndOutputs == numOperands)
          return {};
        SmallVector<OpOperand *> result;
        result.reserve(numOperands - numInputsAndOutputs);
        llvm::transform(
          this->getOperation()->getOpOperands()
            .drop_front(numInputsAndOutputs),
          std::back_inserter(result),
          [](OpOperand &opOperand) {return &opOperand;});
        return result;
      }]
    >,
    //===------------------------------------------------------------------===//
    // Rewriting-related interface methods.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      /*desc=*/[{
        Tiling consumer's operands, given a fused op (a fused vserion of `this`),
        a result offset, a operand 
      }],
      /*retTy=*/"LogicalResult",
      /*methodName=*/"isValidTiling",
      /*args=*/(ins "Operation *": $tiled),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return success();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return whether a producer op is valid with this op (as a consumer). 
      }],
      /*retTy=*/"LogicalResult",
      /*methodName=*/"isValidTiledProducerOp",
      /*args=*/(ins "Operation *": $fusedProducer, 
                    "unsigned": $consumerOperandNumber),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return success();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return whether fused consumer ops are valid with this op (as an unfused producer) 
        through this function. 
        If so, modify it whether it is needed and return a success. 
        If not, directly return a failure. 
      }],
      /*retTy=*/"LogicalResult",
      /*methodName=*/"makeValidTiledConsumerOps",
      /*args=*/(ins "OpBuilder &":$b, "Operation *": $fusedProducer, 
                    "unsigned": $producerResultNumber),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return success();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return whether result is loop invariant during tiling
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isResultLoopInvariant",
      /*args=*/(ins "int64_t": $number, "bool": $hasOneOrZeroUse, "bool": $allParallel),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return false;
      }]
    >,
    //===------------------------------------------------------------------===//
    // Other interface methods.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      /*desc=*/[{
        Return the indexing maps attribute within the current operation.
      }],
      /*retTy=*/"ArrayAttr",
      /*methodName=*/"getIndexingMaps"
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the indexing maps within the current operation.
      }],
      /*retTy=*/"SmallVector<AffineMap>",
      /*methodName=*/"getIndexingMapsArray",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        auto range = $_op.getIndexingMaps()
          .template getAsValueRange<AffineMapAttr>();
        return {range.begin(), range.end()};
      }]
    >,
    //===------------------------------------------------------------------===//
    // Other static interface methods.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      /*desc=*/[{
        Clone the current operation with the given location and operands. This
        is used to abstract away the optional underlying region creation. This
        does not change the balance between input, output_buffer and
        init_tensors operands.
      }],
      /*retTy=*/"Operation *",
      /*methodName=*/"clone",
      (ins "OpBuilder &":$b, "Location":$loc, "TypeRange":$resultTypes,
           "ValueRange":$operands),
      [{
        IRMapping bvm;
        OperationState state(
          loc, ConcreteOp::getOperationName(), operands, resultTypes,
          $_op->getAttrs());
        for (Region &r : $_op->getRegions())
          r.cloneInto(state.addRegion(), bvm);
        return b.create(state);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Clone the current operation with the given location and operands but
        leave the regions empty. This is used to abstract away the optional
        underlying region creation. This does not change the balance between
        input, output_buffer and init_tensors operands.
      }],
      /*retTy=*/"Operation *",
      /*methodName=*/"cloneWithoutRegions",
      (ins "OpBuilder &":$b, "Location":$loc, "TypeRange":$resultTypes,
           "ValueRange":$operands),
      [{
        OperationState state(
          loc, ConcreteOp::getOperationName(), operands, resultTypes,
          $_op->getAttrs());
        for (size_t cnt = 0, e = $_op->getNumRegions(); cnt < e; ++cnt)
          state.addRegion();
        return b.create(state);
      }]
    >
  ];

  let cppNamespace = "::mlir::linalg_ext";

  let extraClassDeclaration = [{
    /// Returns the value that expresses the shape of the output in terms of
    /// shape of the input operands where possible.
    LogicalResult reifyResultShapes(OpBuilder &b,
        mlir::ReifiedRankedShapedTypeDims &reifiedReturnShapes);

    //========================================================================//
    // Helper functions to mutate the `operand_segment_sizes` attribute.
    // These are useful when cloning and changing operand types.
    //========================================================================//
    void setNumInputs(unsigned num) { setOperandSegmentAt(0, num); }
    void setNumOutputBuffers(unsigned num) { setOperandSegmentAt(1, num); }

    private:
    void setOperandSegmentAt(unsigned idx, unsigned val) {
      auto attr = cast<DenseIntElementsAttr>(
          (*this)->getAttr("operand_segment_sizes"));
      unsigned i = 0;
      auto newAttr = attr.mapValues(IntegerType::get(getContext(), 32),
        [&](const APInt &v) { return (i++ == idx) ? APInt(32, val) : v; });
      getOperation()->setAttr("operand_segment_sizes", newAttr);
    }
  }];

  let verify = [{ return detail::verifyLinalgExtOpInterface($_op); }];
}


#endif // BYTEIR_DIALECT_LINALG_IR_LINALG_EXT_INTERFACES