//===- builder.h ----------------------------------------------*--- C++ -*-===//
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

#pragma once

#include "brt/core/ir/ir.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h"
#include <memory>
#include <string>
#include <tuple>
#include <vector>

// forwarding
namespace brt {
namespace ir {

// forwarding
struct ByREBuilderStructImpl;

class ByREBuilder {
public:
  using TypeAndArgAttrsPack =
      std::tuple<mlir::Type, mlir::byre::EntryFuncArgType, std::string>;

  ByREBuilder();

  ~ByREBuilder();

  mlir::func::FuncOp
  CreateEntryPointFuncSignature(const std::string &func_name,
                                // array of (type, arg type, arg name)
                                const std::vector<TypeAndArgAttrsPack> &types);

  // return a ModuleOp
  mlir::ModuleOp GetModuleOp();

  // return a MLIRContext
  mlir::MLIRContext *GetMLIRContext();

  mlir::Block *GetEntryPointFuncBodyBlock();

  void RecordOperation(mlir::Operation *);

  mlir::Operation *GetRecordOperation();

private:
  std::unique_ptr<ByREBuilderStructImpl> impl_;
};

} // namespace ir
} // namespace brt
