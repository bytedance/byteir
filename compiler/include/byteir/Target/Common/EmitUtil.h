//===- EmitUtil.h ---------- ----------------------------------------------===//
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

#ifndef BYTEIR_TARGET_COMMON_EMITUTIL_H
#define BYTEIR_TARGET_COMMON_EMITUTIL_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

namespace byteir {

/// Convenience functions to produce interleaved output with functions returning
/// a LogicalResult. This is different than those in STLExtras as functions used
/// on each element doesn't return a string.
template <typename ForwardIterator, typename UnaryFunctor,
          typename NullaryFunctor>
inline mlir::LogicalResult
interleaveWithError(ForwardIterator begin, ForwardIterator end,
                    UnaryFunctor eachFn, NullaryFunctor betweenFn) {
  if (begin == end)
    return mlir::success();
  if (mlir::failed(eachFn(*begin)))
    return mlir::failure();
  ++begin;
  for (; begin != end; ++begin) {
    betweenFn();
    if (mlir::failed(eachFn(*begin)))
      return mlir::failure();
  }
  return mlir::success();
}

template <typename Container, typename UnaryFunctor, typename NullaryFunctor>
inline mlir::LogicalResult interleaveWithError(const Container &c,
                                               UnaryFunctor eachFn,
                                               NullaryFunctor betweenFn) {
  return interleaveWithError(c.begin(), c.end(), eachFn, betweenFn);
}

template <typename Container, typename UnaryFunctor>
inline mlir::LogicalResult interleaveCommaWithError(const Container &c,
                                                    llvm::raw_ostream &os,
                                                    UnaryFunctor eachFn) {
  return interleaveWithError(c.begin(), c.end(), eachFn, [&]() { os << ", "; });
}

} // namespace byteir

#endif // BYTEIR_TARGET_COMMON_EMITUTIL_H