//===-- LinalgExtBase.td --------------------------------------------------===//
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
// Some code comes from LinalgExtBase.td in IREE project
// Original license:
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_LINALG_IR_LINALG_EXT_BASE
#define BYTEIR_DIALECT_LINALG_IR_LINALG_EXT_BASE

include "mlir/IR/OpBase.td"

def LinalgExt_Dialect : Dialect {
  let name = "linalg_ext";

  let summary = "Extension for Linalg dialect";

  let cppNamespace = "::mlir::linalg_ext";
  let dependentDialects = [
    "arith::ArithDialect",
    "affine::AffineDialect",
    "math::MathDialect",
    "memref::MemRefDialect",
    "tensor::TensorDialect",
  ];

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Type definitions
//===----------------------------------------------------------------------===//

class RankedTensorOrMemRefOf<list<Type> allowedTypes> :
  ShapedContainerType<allowedTypes,
      Or<[IsMemRefTypePred, And<[IsTensorTypePred, HasRankPred]>]>,
  "ranked tensor or memref", "::mlir::ShapedType">;

def AnyRankedTensorOrMemRefType : RankedTensorOrMemRefOf<[AnyType]>;

#endif // BYTEIR_DIALECT_LINALG_IR_LINALG_EXT_BASE