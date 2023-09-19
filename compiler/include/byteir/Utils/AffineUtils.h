//===- AffineUtils.h ----------------------------------------------- C++---===//
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

#ifndef BYTEIR_UTILS_AFFINEUTILS_H
#define BYTEIR_UTILS_AFFINEUTILS_H

#include "mlir/IR/AffineMap.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {

/// find iteration index through dim and inversePermutation
/// E.g. if affineMap = (d0, d1, d2)-> (d0, d2), dim = 1
/// return 2  (from d2)
FailureOr<unsigned> getIterAxisFromDim(AffineMap affineMap, unsigned dimIndex);

AffineMap getFlattenAffineMap(mlir::MLIRContext *,
                              ArrayRef<int64_t> staticShape);

/// return a `numDims` affineMap without dim `skips`
/// E.g. if numDims = 3, skips = {1, 2}
/// return affineMap = (d0, d1, d2)-> (d0)
AffineMap getMultiDimIdentityMapWithSkips(unsigned numDims,
                                          ArrayRef<int64_t> skips,
                                          MLIRContext *context);

/// return a `numDims` affineMap with only dim `targets`
/// E.g. if numDims = 3, targets = {1, 2}
/// return affineMap = (d0, d1, d2)-> (d1, d2)
AffineMap getMultiDimIdentityMapWithTargets(unsigned numDims,
                                            ArrayRef<int64_t> targets,
                                            MLIRContext *context);

/// Returns true if the AffineMap represents a subset (i.e. a projection) of a
/// symbol-less permutation map. It allows projected permutation maps with
/// constant result expressions.
bool isProjectedPermutationAndAllowConst(AffineMap map);

} // namespace mlir

#endif // BYTEIR_UTILS_AFFINEUTILS_H
