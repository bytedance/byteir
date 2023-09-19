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
#include "llvm/ADT/SmallSet.h"

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

AffineMap mlir::getFlattenAffineMap(mlir::MLIRContext *ctx,
                                    ArrayRef<int64_t> staticShape) {
  SmallVector<int64_t> strides;
  unsigned numDim = staticShape.size();
  strides.reserve(numDim);
  int64_t prod = 1;
  for (auto it = staticShape.rbegin(); it != staticShape.rend(); ++it) {
    strides.push_back(prod);
    prod *= *it;
  }
  strides = llvm::to_vector(llvm::reverse(strides));

  AffineExpr result = staticShape[0] > 1 ? getAffineDimExpr(0, ctx) * strides[0]
                                         : getAffineDimExpr(0, ctx) * 0;
  for (unsigned i = 1; i < numDim; ++i) {
    if (staticShape[i] > 1) {
      AffineExpr x = getAffineDimExpr(i, ctx);
      result = result + x * strides[i];
    }
  }
  SmallVector<AffineExpr, 2> results;
  results.push_back(result);
  return AffineMap::get(numDim, 0, results, ctx);
}

AffineMap mlir::getMultiDimIdentityMapWithSkips(unsigned numDims,
                                                ArrayRef<int64_t> skips,
                                                MLIRContext *context) {
  llvm::SmallSet<int64_t, 4> skipSet;
  skipSet.insert(skips.begin(), skips.end());
  SmallVector<AffineExpr, 4> dimExprs;
  dimExprs.reserve(numDims);
  for (unsigned i = 0; i < numDims; ++i) {
    if (skipSet.contains(i)) {
      continue;
    }
    dimExprs.push_back(mlir::getAffineDimExpr(i, context));
  }
  return AffineMap::get(/*dimCount=*/numDims, /*symbolCount=*/0, dimExprs,
                        context);
}

AffineMap mlir::getMultiDimIdentityMapWithTargets(unsigned numDims,
                                                  ArrayRef<int64_t> targets,
                                                  MLIRContext *context) {
  AffineMap result =
      AffineMap::get(/*dimCount=*/numDims, /*symbolCount=*/0, context);
  int64_t pos = 0;
  for (int64_t t : targets) {
    result = result.insertResult(getAffineDimExpr(t, context), pos);
    pos += 1;
  }
  return result;
}

bool mlir::isProjectedPermutationAndAllowConst(AffineMap map) {
  if (map.getNumSymbols() > 0)
    return false;

  if (map.getNumResults() > map.getNumInputs())
    return false;

  SmallVector<bool, 8> seen(map.getNumInputs(), false);
  for (auto expr : map.getResults()) {
    if (auto dim = expr.dyn_cast<AffineDimExpr>()) {
      if (seen[dim.getPosition()])
        return false;
      seen[dim.getPosition()] = true;
    } else {
      if (!expr.isa<AffineConstantExpr>())
        return false;
    }
  }

  return true;
}
