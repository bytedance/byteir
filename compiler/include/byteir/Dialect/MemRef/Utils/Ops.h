//===- Ops.h --------------------------------------------------------------===//
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

#ifndef BYTEIR_DIALECT_MEMREF_UTILS_OPS_H
#define BYTEIR_DIALECT_MEMREF_UTILS_OPS_H

namespace mlir {

namespace memref {
class SubViewOp;
}

// To check whether two subviews are overlapped or not, we can use a
// conservative approach. This means that when the function returns true, we can
// be certain that the two subviews are not overlapped. However, if the function
// returns false, we cannot say for certain whether the subviews are overlapped
// or not.
bool doSubViewsConservativelyNotOverlap(memref::SubViewOp lhs,
                                        memref::SubViewOp rhs);

} // namespace mlir

#endif // BYTEIR_DIALECT_MEMREF_UTILS_OPS_H