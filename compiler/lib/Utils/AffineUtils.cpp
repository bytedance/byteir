//===- AffineUtils.cpp ----------------------------------------------------===//
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

#include "byteir/Utils/AffineUtils.h"
#include "byteir/Utils/Utils.h"

using namespace mlir;

namespace {

// a high-complexity bruteforce full search
FailureOr<unsigned> getIterAxisFromDimForwardFullSearch(AffineMap affineMap,
                                                        unsigned dimIndex) {
  SmallVector<unsigned> iterAxes;
  for (unsigned i = 0; i < affineMap.getNumInputs(); ++i) {
    auto composed =
        affineMap.compose(createOneHot(affineMap.getNumInputs(), i));

    auto dims = getAllIndicesForNonZeros(composed);

    // no support all-to-1 or non mapping
    if (dims.size() == 1 && dims[0] == dimIndex) {
      iterAxes.push_back(i);
    }
  }

  // no support all-to-1 or non mapping
  if (iterAxes.size() != 1) {
    return failure();
  }
  return iterAxes[0];
}

} // namespace

/**
 * find iteration index through dim and inverseMap
 * E.g. if affineMap = (d0, d1, d2)-> (d0, d2), dim = 1
 * Then invMap = (d0, d1)->(d0, 0, d1)
 *      oneHot = (0, 1)
 *      invComposed = (0, 0, 1)
 *      iterAxis = 2
 **/
FailureOr<unsigned> mlir::getIterAxisFromDim(AffineMap affineMap,
                                             unsigned dimIndex) {
  AffineMap invMap;
  if (affineMap.isProjectedPermutation()) {
    invMap = inverseAndBroadcastProjectedPermutation(affineMap);
  } else if (affineMap.isPermutation()) {
    invMap = inversePermutation(affineMap);
  } else {
    // if no invMap, we do forward full search
    return getIterAxisFromDimForwardFullSearch(affineMap, dimIndex);
  }

  if (invMap.isEmpty())
    return failure();

  auto invComposed =
      invMap.compose(createOneHot(invMap.getNumInputs(), dimIndex));
  auto iterAxes = getAllIndicesForNonZeros(invComposed);
  // no support all-to-1 or non mapping
  if (iterAxes.size() != 1) {
    return failure();
  }
  return iterAxes[0];
}
