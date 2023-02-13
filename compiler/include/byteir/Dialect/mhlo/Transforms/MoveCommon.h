//===- MoveCommon.h ------------------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_DIALECT_MHLO_TRANSFORMS_MOVECOMMON_H
#define BYTEIR_DIALECT_MHLO_TRANSFORMS_MOVECOMMON_H

#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
class MLIRContext;

// common Move patterns template for some general cases

template <typename OTy>
struct HloMoveDownPattern : public OpRewritePattern<OTy> {
  HloMoveDownPattern(MLIRContext *ctx,
                     const llvm::DenseSet<llvm::StringRef> &blocker,
                     bool supportAllMultiUser = false,
                     bool supportMultiUser = false)
      : OpRewritePattern<OTy>(ctx), blockers(blocker),
        allMultiUser(supportAllMultiUser), multiUser(supportMultiUser) {}

  const llvm::DenseSet<llvm::StringRef> &blockers;
  bool allMultiUser; // allow transposed result used in multiple users, all of
                     // them must be legal
  bool multiUser; // allow transposed result used in multiple users, including
                  // some illegal ops

  /*
  //  multiUser  = false,  AllMultiUser = false
  case 1
            S
            |
          OTy
          /   \
        A     B
  ==> no transformation
  */

  /*
  //  multiUser  = true,  AllMultiUser = false
  case 1
            S
            |
            OTy
          /   \
          A     B
  ==>        S
            /  \
          A    B
          |    |
          OTy  OTy
  case 2
            S
            |
            OTy
          /  |  \
          A   B   IllegalC
  ==>         S
            / |  \
          A  B  OTy
          |  |   |
          OTy OTy IllegalC
  */

  /*
  //  AllMultiUser = true,  multiUser = true/false/don't care
  case 1
            S
            |
            OTy
          /  |  \
          A   B   IllegalC
  ==> no transformation
  */
};

template <typename OTy> struct HloMoveUpPattern : public OpRewritePattern<OTy> {
  HloMoveUpPattern(MLIRContext *ctx,
                   const llvm::DenseSet<llvm::StringRef> &blocker,
                   bool supportMultiInput = false)
      : OpRewritePattern<OTy>(ctx), blockers(blocker),
        multiInput(supportMultiInput) {}

  const llvm::DenseSet<llvm::StringRef> &blockers;
  bool multiInput; // allow producer of transpose has multiple inputs
  /*
  //  multiInput  = false
  case 1
           S1    S2
            \   /
              A
              |
              OTy
    ==> no transformation
  */

  /*
  //  multiInput  = true
  case 1
             S1    S2
              \   /
                A
                |
                OTy
  =>
            S1    S2
            |     |
            OTy   OTy
              \  /
                A
  */
};

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_TRANSFORMS_MOVECOMMON_H