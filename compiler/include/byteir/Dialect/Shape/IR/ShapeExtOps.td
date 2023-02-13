//===-- ShapeExtOps.td -------------------------------------------------------===//
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

#ifndef BYTEIR_DIALECT_SHAPE_IR_SHAPE_EXT_OPS
#define BYTEIR_DIALECT_SHAPE_IR_SHAPE_EXT_OPS

include "byteir/Dialect/Shape/IR/ShapeExtBase.td"

// Base class for the operation in this dialect
class ShapeExt_Op<string mnemonic, list<Trait> traits = []> :
    Op<ShapeExtDialect, mnemonic, traits>;

def ShapeExt_TieOp: ShapeExt_Op<"tie"> {
  let summary = "binding a set of multidim shape.size with a tensor value";
  let description = [{
    Binding a set of multidim shape.size with a tensor value.
  }];
  let arguments = (ins
    AnyTensor:$value,
    Arg<Variadic<Shape_SizeOrIndexType>, "">:$dims
  );
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}


def ShapeExt_MeetOp: ShapeExt_Op<"meet"> {
  let summary = "an alternative of shape.meet to support index type";
  let arguments = (ins
    Shape_SizeOrIndexType:$arg0,
    Shape_SizeOrIndexType:$arg1
  );
}

#endif // BYTEIR_DIALECT_SHAPE_IR_SHAPE_EXT_OPS