//===- PatternMatch.cpp ---------------------------------- -*- C++ ------*-===//
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

#include "byteir/Utils/PatternMatch.h"
#include "mlir/Dialect/PDL/IR/PDL.h"

using namespace mlir;
using namespace llvm;

namespace {
class PDLPatternHooksInterface
    : public DialectInterface::Base<PDLPatternHooksInterface> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PDLPatternHooksInterface);

  PDLPatternHooksInterface(Dialect *dialect) : Base(dialect) {}

  bool add(StringRef name, PDLConstraintFunction constraintFn, bool override);
  bool add(StringRef name, PDLRewriteFunction rewriteFn, bool override);

  void apply(PDLPatternModule &pdlModule) const;

  static PDLPatternHooksInterface *get(MLIRContext *ctx);

private:
  llvm::StringMap<PDLConstraintFunction> constraintFunctions;
  llvm::StringMap<PDLRewriteFunction> rewriteFunctions;
};

bool PDLPatternHooksInterface::add(StringRef name,
                                   PDLConstraintFunction constraintFn,
                                   bool override) {
  if (override) {
    constraintFunctions.insert_or_assign(name, std::move(constraintFn));
    return true;
  } else {
    return constraintFunctions.try_emplace(name, std::move(constraintFn))
        .second;
  }
}

bool PDLPatternHooksInterface::add(StringRef name, PDLRewriteFunction rewriteFn,
                                   bool override) {
  if (override) {
    rewriteFunctions.insert_or_assign(name, std::move(rewriteFn));
    return true;
  } else {
    return rewriteFunctions.try_emplace(name, std::move(rewriteFn)).second;
  }
}

void PDLPatternHooksInterface::apply(PDLPatternModule &pdlPattern) const {
  for (auto &&it : constraintFunctions) {
    pdlPattern.registerConstraintFunction(it.first(), it.second);
  }
  for (auto &&it : rewriteFunctions) {
    pdlPattern.registerRewriteFunction(it.first(), it.second);
  }
}
} // namespace

PDLPatternHooksInterface *PDLPatternHooksInterface::get(MLIRContext *ctx) {
  auto dialect = ctx->getOrLoadDialect<pdl::PDLDialect>();
  if (!dialect)
    return nullptr;

  return dialect->getRegisteredInterface<PDLPatternHooksInterface>();
}

bool mlir::registerPDLConstraintFunction(MLIRContext *ctx, StringRef name,
                                         PDLConstraintFunction constraintFn,
                                         bool override) {
  auto iface = PDLPatternHooksInterface::get(ctx);
  if (!iface)
    return false;

  return iface->add(name, std::move(constraintFn), override);
}

bool mlir::registerPDLRewriteFunction(MLIRContext *ctx, StringRef name,
                                      PDLRewriteFunction rewriteFn,
                                      bool override) {
  auto iface = PDLPatternHooksInterface::get(ctx);
  if (!iface)
    return false;

  return iface->add(name, std::move(rewriteFn), override);
}

void mlir::applyPDLPatternHooks(PDLPatternModule &pdlPattern) {
  auto ctx = pdlPattern.getModule().getContext();
  auto iface = PDLPatternHooksInterface::get(ctx);
  if (!iface)
    return;

  iface->apply(pdlPattern);
}

void mlir::registerPDLPatternHooksInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, pdl::PDLDialect *dialect) {
    dialect->addInterfaces<PDLPatternHooksInterface>();
  });
}