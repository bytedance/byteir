//===-- ShapeExtBase.td ---------------------------------------------------===//
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

#ifndef BYTEIR_DIALECT_SHAPE_IR_SHAPE_EXT_BASE
#define BYTEIR_DIALECT_SHAPE_IR_SHAPE_EXT_BASE

include "mlir/Dialect/Shape/IR/ShapeBase.td"

def ShapeExtDialect : Dialect {
  let name = "shape_ext";

  let summary = "Extension for shape dialect";

  let cppNamespace = "::mlir::shape_ext";
  let dependentDialects = ["arith::ArithDialect", "tensor::TensorDialect", "shape::ShapeDialect"];

  // let useDefaultTypePrinterParser = 1;
  // let hasConstantMaterializer = 1;
  // let hasOperationAttrVerify = 1;
}

#endif // BYTEIR_DIALECT_SHAPE_IR_SHAPE_EXT_BASE