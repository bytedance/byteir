//===- ir.h ---------------------------------------------------*--- C++ -*-===//
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

#include "brt/core/common/status.h"
#include "brt/core/ir/graph_info.h"
#include <functional>
#include <memory>
#include <string>

// forwarding
namespace mlir {
class BlockArgument;
class Operation;
class WalkResult;
class DictionaryAttr;

namespace byre {
class ByreOp;
} // namespace byre

} // namespace mlir

namespace brt {
namespace ir {

/**
 * IRHandle holds Runtime IR handle
 * IRHandle is a abstract calls
 */

class IRHandle {
public:
  IRHandle() {}

  virtual ~IRHandle(){};

  /**
   * Initialize function initialize IRHandle related state.
   * It is assumed to be once called once in the beginning.
   */
  virtual common::Status Initialize() = 0;

  /**
   * Load a model from a given file `url` and corresonding format `fmt`
   * The supported format is "byre" for now.
   */
  virtual common::Status Load(const std::string &url,
                              const std::string &fmt) = 0;

  /*
   * Load a model from an in-memory IR
   */
  virtual common::Status LoadFromMemory(const void *,
                                        const std::string &fmt) = 0;

  /*
   * dump IR for debug
   */
  virtual void dump() = 0;

  // Init GraphInfo
  virtual void InitGraphInfoNameAndArgOffset(GraphInfo &info) = 0;
};

// forwarding
struct ByREHandleImpl;

/**
 * IRHandle holds Runtime IR handle
 * IRHandle is a abstract calls
 */
class ByREHandle : public IRHandle {
public:
  ByREHandle();

  virtual ~ByREHandle();

  /**
   * InitializeIR function initialize IR related state.
   * It is assumed to be once called once in the beginning.
   */
  common::Status Initialize() override;

  /**
   * Load a model from a given file `url` and corresonding format `fmt`
   * The supported format is "byre" for now.
   */
  common::Status Load(const std::string &url, const std::string &fmt) override;

  /*
   * Load a model from an in-memory IR
   */
  common::Status LoadFromMemory(const void *, const std::string &fmt) override;

  /*
   * dump IR for debug
   */
  void dump() override;

  /**
   * IterateNode iterates node of graph, aka mlir::Operation* of a IRHandle
   */
  // TODO change Operation to ByreOp after adding ByreOpInterface
  // common::Status IterateNode(std::function<void(mlir::Operation*)> func);
  common::Status
  IterateNode(std::function<mlir::WalkResult(mlir::Operation *)> func);

  /**
   * IterateEntryFuncArg iterates args of EntryFunc, aka BlockArgument of
   * EntryFunc
   */
  common::Status IterateEntryFuncArg(
      std::function<common::Status(mlir::BlockArgument, mlir::DictionaryAttr)>
          func);

  // Init GraphInfo
  void InitGraphInfoNameAndArgOffset(GraphInfo &info) override;

  // Return an OpNmae of ByreOp
  // Key is per combination of OpKind
  static std::string GetOpKind(mlir::byre::ByreOp op);

  // Return a Key of ByreOp
  // Key is per combination of OpKind, and Arg Types
  static std::string GetKey(mlir::byre::ByreOp op);

  // Return a OpUID of ByreOp
  // OpUID is per instance of Op
  static std::string GetOpUID(mlir::byre::ByreOp op);

  // Return path to .mlir file
  std::string &GetIRPath();

  ByREHandleImpl &getImpl() { return *impl_; }

private:
  std::unique_ptr<ByREHandleImpl> impl_;
  std::string ir_path_;
};

} // namespace ir
} // namespace brt
