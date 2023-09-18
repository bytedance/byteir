//===- ShardingInterfaces.td -------------------------------*- tablegen -*-===//
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

#ifndef BYTEIR_DIALECT_MESH_INTERFACES_SHARDING_INTERFACE
#define BYTEIR_DIALECT_MESH_INTERFACES_SHARDING_INTERFACE

include "mlir/IR/OpBase.td"

def ShardingInterface : OpInterface<"ShardingInterface"> {
    let description = [{
        Interface for allowing operations to expose information needed to
        shard them.
    }];
    let cppNamespace = "::mlir::mesh";

    let methods = [
      InterfaceMethod<
        /*desc=*/[{
          Returns a list of iterator types that describe the number of loops.
        }],
        /*retType=*/"SmallVector<::mlir::mesh::ShardingIteratorType>",
        /*methodName=*/"getLoopIteratorTypes",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/"return {};"
      >,
      InterfaceMethod<
        /*desc=*/[{
          Return the indexing maps attribute within the current operation.
        }],
        /*retTy=*/"SmallVector<AffineMap>",
        /*methodName=*/"getIndexingMaps",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/"return {};"
      >,
      InterfaceMethod<
        /*desc=*/[{
          Given that certain operands or results of the operation may be
          annotated, this method leverages this information to deduce how the operation
          should be sharded.
        }],
        /*retTy=*/"FailureOr<ShardingOption>",
        /*methodName=*/"getShardingOption",
        /*args=*/(ins
          "OpBuilder &":$b
        ),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return detail::defaultGetShardingOption(
            $_op.getOperation(), b);
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{
          Based on a given ShardingOption, this method annotates those
          operands and results which previously lacked sharding annotations.
        }],
        /*retTy=*/"LogicalResult",
        /*methodName=*/"setShardingAnnotations",
        /*args=*/(ins
          "OpBuilder &":$b
        ),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return detail::defaultSetShardingAnnotations(
            $_op.getOperation(), b);
        }]
      >
    ];

    let extraClassDeclaration = [{
      LogicalResult verifyShardingInterfaceImpl();

      void printLoopTypesAndIndexingMaps(raw_ostream &os);
    }];
}


#endif // BYTEIR_DIALECT_MESH_INTERFACES_SHARDING_INTERFACE
