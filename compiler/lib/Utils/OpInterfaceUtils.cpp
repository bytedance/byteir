//===- OpInterfaceUtils.cpp ------------------------------ -*- C++ ------*-===//
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

#include "byteir/Utils/OpInterfaceUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace llvm;

namespace {
struct ExtensionRegistry {
  struct Extension final : public DialectExtensionBase {
    using ImplT = std::function<void(MLIRContext *ctx)>;
    using CtorParam = std::pair<ImplT, StringRef>;

    Extension(const CtorParam &param)
        : DialectExtensionBase(ArrayRef<StringRef>{param.second}),
          impl(param.first) {}

    void apply(MLIRContext *context,
               MutableArrayRef<Dialect *> /* dialects */) const override {
      if (enableOpInterfaceExtensions) {
        impl(context);
      }
    }

    std::unique_ptr<DialectExtensionBase> clone() const override {
      return std::make_unique<Extension>(*this);
    }

    ImplT impl;
  };

  void insert(Extension::ImplT extensionFn, StringRef dialectName) {
    ctorParams.push_back({std::move(extensionFn), dialectName});
  }

  void apply(DialectRegistry &registry) {
    for (auto &&param : ctorParams) {
      registry.addExtension(std::make_unique<Extension>(param));
    }
  }

  static ExtensionRegistry &inst();

private:
  static llvm::cl::opt<bool> enableOpInterfaceExtensions;

  SmallVector<Extension::CtorParam> ctorParams;
};

ExtensionRegistry &ExtensionRegistry::inst() {
  static ExtensionRegistry inst;
  return inst;
}

llvm::cl::opt<bool> ExtensionRegistry::enableOpInterfaceExtensions(
    "enable-op-interface-extensions",
    llvm::cl::desc("Enable op interface extensions, this would override "
                   "some implementations of op interface"),
    llvm::cl::init(true));
}; // namespace

void mlir::detail::addOpInterfaceExtension(
    std::function<void(MLIRContext *ctx)> extensionFn,
    llvm::StringRef dialectName) {
  ExtensionRegistry::inst().insert(std::move(extensionFn), dialectName);
}

void mlir::registerOpInterfaceExtensions(DialectRegistry &registry) {
  ExtensionRegistry::inst().apply(registry);
}
