//===- session.h ----------------------------------------------*--- C++ -*-===//
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
#include "brt/core/framework/dtype.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace brt {

// forward decl
class ExecutionPlan;
class ExecutionProvider;
class IAllocator;
class OpKernelInfo;
class RequestContext;
class WorkQueue;

namespace ir {
class IRHandle;
}

/**
 * Session is a base class for the runtime.
 * Each Session holds one model.
 *
 * Session can be derived to either InferenceSession and TraingSession
 * for different purposes.
 */

class Session {
public:
  // TODO add Session options later
  Session();

  virtual ~Session();

  // load a model
  common::Status Load(const std::string &url, const std::string &fmt);

  /*
   * Load a model from an in-memory IR
   */
  common::Status LoadFromMemory(const void *, const std::string &fmt);

  // Return a Weight's pointer
  void *GetWeightAsyncValue(size_t);

  // Load a file to initialize weights
  common::Status LoadWeights(const std::string &url, const std::string &fmt);

  // Create a new RequestContext
  // Note: request_ctx would take the ownership of the \p work_queue if it is
  // not nullptr
  common::Status NewRequestContext(std::unique_ptr<RequestContext> *request_ctx,
                                   WorkQueue *work_queue = nullptr);

  // intialize a weight
  common::Status InitializeWeight(size_t id, const void *ptr);

  // Return Tensor Index from a string
  // Return -1 if not found
  int GetGraphArgOffset(const std::string &name);

  // Return the first tensor index which the arg_offset's tensor alias to
  // Return -1 if not found
  int GetGraphArgAliasOffset(const size_t arg_offset);

  // Run a model for a given RequestContext
  common::Status Run(RequestContext &request_ctx);

  // Do some cleanup for RequestContext
  void Cleanup(RequestContext &request_ctx) noexcept;

  // Return arg number
  size_t GetArgNum();

  // Return arg number
  size_t GetWeightNum();

  const std::vector<std::string> &GetWeightNames();
  const std::vector<std::string> &GetInputNames();
  const std::vector<std::string> &GetOutputNames();
  const std::vector<size_t> &GetWeightArgOffsets();
  const std::vector<size_t> &GetInputArgOffsets();
  const std::vector<size_t> &GetOutputArgOffsets();

  // Return a Shape from a Tensor Index
  const std::vector<int64_t> GetStaticShape(size_t id);

  // Return dtype from a Tensor Index
  DTypeEnum GetDType(size_t id);

  /**
   * Add an ExecutionProivder into a Session
   */
  common::Status
  AddExecutionProvider(std::unique_ptr<ExecutionProvider> provider);

  /**
   * Add an Allocator into a Session
   */
  common::Status AddAllocator(std::unique_ptr<IAllocator> allocator);

  // Return an Allocator
  IAllocator *GetAllocator(const std::string &key);

protected:
  // hold a set of execution providers
  std::vector<std::unique_ptr<ExecutionProvider>> exec_providers_;

  std::unordered_map<std::string, std::unique_ptr<IAllocator>> allocators_;

  // hold an execution plan
  // TODO: confirm this for TrainingSession
  std::unique_ptr<ExecutionPlan> execution_plan_;

  // hold an IR handle
  std::unique_ptr<brt::ir::IRHandle> ir_handle_;
};

} // namespace brt
