//===- OFCheckNonLowered.cpp ----------------------------------------------===//
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

#include "onnx-frontend/src/Conversion/OFCheckNonLowered.hpp"
#include "third_party/onnx-mlir/src/Dialect/ONNX/ONNXDialect.hpp"
#include "third_party/onnx-mlir/src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "llvm/ADT/Twine.h"

using namespace mlir;

namespace {

struct OFCheckNonLoweredPass
    : public onnx_frontend::OFCheckNonLoweredBase<OFCheckNonLoweredPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OFCheckNonLoweredPass)

  OFCheckNonLoweredPass() = default;

  void runOnOperation() final {
    func::FuncOp func = getOperation();
    bool onnxFound = false;
    func.walk([&](Operation *op) {
      // Check if the op is an ONNX op.
      if (isa<ONNXDialect>(op->getDialect())) {
        llvm::Twine msg(op->getName().getStringRef() +
                        ": ONNX op is not lowered");
        emitWarning(op->getLoc(), msg);
        onnxFound = true;
      }
    });
    if (!onnxFound) {
      llvm::Twine msg("All ONNX ops are lowered");
      emitRemark(func.getLoc(), msg);
    } else {
      llvm::Twine msg("Please lower all ONNX ops");
      emitError(func.getLoc(), msg);
      return signalPassFailure();
    }
  }
};

} // namespace

namespace onnx_frontend {
std::unique_ptr<mlir::Pass> createOFCheckNonLoweredPass() {
  return std::make_unique<OFCheckNonLoweredPass>();
}
} // namespace onnx_frontend