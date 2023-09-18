//===- InferDTensorInterface.td ----------------------------*- tablegen -*-===//
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

#ifndef BYTEIR_DIALECT_MESH_INTERFACES_INFER_DTENSOR_INTERFACE
#define BYTEIR_DIALECT_MESH_INTERFACES_INFER_DTENSOR_INTERFACE

include "mlir/IR/OpBase.td"

def InferDTensorInterface : OpInterface<"InferDTensorInterface"> {
    let description = [{
        InferDTensorInterface is an interface in MLIR that provides a mechanism
        to infer the distributed tensor encoding of an operation's outputs based
        on the encoding of its operands.  This interface is particularly useful
        for operations that transform tensors in a distributed setting, e.g. ccl
        operations in mesh dialect.
    }];
    let cppNamespace = "::mlir::mesh";

    let methods = [
      StaticInterfaceMethod<
      /*desc=*/[{
        This method is responsible for inferring the distributed tensor encoding
        of the operation's outputs based on the encoding of its operands.
      }],
      /*retTy=*/"::mlir::LogicalResult",
      /*methodName=*/"inferReturnTypes",
      /*args=*/(ins "::mlir::MLIRContext *":$context,
                    "::std::optional<::mlir::Location>":$location,
                    "::mlir::ValueRange":$operands,
                    "::mlir::DictionaryAttr":$attributes,
                    "::mlir::OpaqueProperties":$properties,
                    "::mlir::RegionRange":$regions,
                    "::llvm::SmallVectorImpl<::mlir::Type>&":$inferredReturnTypes)
    >
    ];
}


#endif // BYTEIR_DIALECT_MESH_INTERFACES_INFER_DTENSOR_INTERFACE
