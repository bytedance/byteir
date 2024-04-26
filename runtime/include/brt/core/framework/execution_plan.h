//===- execution_plan.h ---------------------------------------*--- C++ -*-===//
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
#include "brt/core/context/execution_context.h"
#include "brt/core/context/execution_frame.h"
#include "brt/core/framework/allocator.h"
#include "brt/core/framework/device_api.h"
#include "brt/core/framework/dtype.h"
#include "brt/core/ir/graph_info.h"
#include "brt/core/ir/ir.h"
#include <memory>
#include <unordered_map>
#include <vector>

namespace brt {

// Forwarding
class ExecutionProvider;
class OpKernel;
class WorkQueue;

/**
 * ExecutionPlan is an abstract class that determines
 * invocation of OpKernels and memory plan.
 *
 * Note ExecutionPlan is shared within a Session across Runs.
 */

class ExecutionPlan {
public:
  ExecutionPlan() {}

  virtual ~ExecutionPlan() = default;

  virtual common::Status ProloguePerSession(
      const std::unordered_map<std::string, std::unique_ptr<IAllocator>>
          &allocators,
      const std::vector<std::unique_ptr<ExecutionProvider>> &providers,
      const Device dev, const DeviceAPI *device_api) = 0;

  virtual common::Status EpiloguePerSession() = 0;

  virtual void CreateWorkQueue(std::unique_ptr<WorkQueue> *wq,
                               int rank = 0) = 0;

  virtual void CreateExecutinFrame(std::unique_ptr<ExecutionFrame> *frame) = 0;

  // Return a Weight's Value
  virtual AsyncValue GetWeightAsyncValue(size_t) = 0;

  // Return a Static ShapeRef
  virtual const Shape GetStaticShape(size_t) = 0;

  virtual DTypeEnum GetDType(size_t) = 0;

  virtual std::string GetSpace(size_t) = 0;

  inline brt::ir::GraphInfo &GetGraphInfo() { return graph_info_; }

  virtual common::Status LoadWeights(const std::string &url,
                                     const std::string &fmt) = 0;

  ExecutionFrame::StateInfo &GetFrameStateInfo() { return frame_state_info_; }

  virtual common::Status ProloguePerFrame(const ExecutionContext &) = 0;
  virtual common::Status EpiloguePerFrame(const ExecutionContext &) = 0;

  virtual common::Status Run(const ExecutionContext &) = 0;

  /*
   * Iterate over all OpKernels and apply callback in execution order
   * \p callback return false to stop iterating
   */
  virtual void IterateOpKernels(std::function<bool(OpKernel *)> callback) = 0;

protected:
  // StateInfo holds info to create state of BRTInferenceFrameInfo
  ExecutionFrame::StateInfo frame_state_info_;

  brt::ir::GraphInfo graph_info_;
};

/**
 * StaticBRTExecutionPlan is the default derived ExecutionPlan in BRT.
 * It invokes OpKernels based on the predefined deserialized graph's order
 * in the compiler.
 *
 * It selects ExecutionProviders based ranking.
 */

class StaticBRTExecutionPlan final : public ExecutionPlan {
public:
  StaticBRTExecutionPlan(brt::ir::ByREHandle &);

  common::Status ProloguePerSession(
      const std::unordered_map<std::string, std::unique_ptr<IAllocator>>
          &allocators,
      const std::vector<std::unique_ptr<ExecutionProvider>> &providers,
      const Device dev, const DeviceAPI *device_api) override;

  common::Status EpiloguePerSession() override;

  void CreateWorkQueue(std::unique_ptr<WorkQueue> *wq, int rank = 0) override;

  void CreateExecutinFrame(std::unique_ptr<ExecutionFrame> *frame) override;

  AsyncValue GetWeightAsyncValue(size_t) override;

  const Shape GetStaticShape(size_t) override;

  DTypeEnum GetDType(size_t) override;

  std::string GetSpace(size_t) override;

  common::Status LoadWeights(const std::string &url,
                             const std::string &fmt) override;

  common::Status ProloguePerFrame(const ExecutionContext &) override;
  common::Status EpiloguePerFrame(const ExecutionContext &) override;

  common::Status Run(const ExecutionContext &) override;

  void IterateOpKernels(std::function<bool(OpKernel *)> callback) override;

private:
  // using std::bind or lambda to push_back kernels and initBeforeRun_
  // store the static order of kernels
  std::vector<std::shared_ptr<OpKernel>> op_kernels_;
  // store the static order for inits
  std::vector<std::shared_ptr<OpKernel>> op_prologue_per_frame_;
  // store the static order for inits
  std::vector<std::shared_ptr<OpKernel>> op_epilogue_per_frame_;

  // TODO find a way to remove this
  brt::ir::ByREHandle &graph_;

  // ConstructInfo holds infomation to construct a BRTInferenceFrameInfo
  BRTInferenceExecutionFrame::ConstructInfo frame_construct_info_;

  // split all op kernels into two set of kernels, for shape deduction and
  // for normal computation, shape deduction ops would be scheduled before
  // intermediate allocation and those ops could run on tensor metadata without
  // concrete tensor data
  std::vector<OpKernel *> shape_op_kernels_;
  std::vector<OpKernel *> compute_op_kernels_;
};

} // namespace brt
