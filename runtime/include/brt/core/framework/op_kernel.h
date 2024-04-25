//===- op_kernel.h --------------------------------------------*--- C++ -*-===//
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
#include "brt/core/framework/event.h"
#include "brt/core/framework/op_kernel_info.h"

/**
 * OpKernel defines an abstract class holding a kernel implementation in
 * 'Kernel'.
 *
 * OpKerenl object is one per node instance, not per kind.
 * Either shape infernece or kernel computation can be implemented as OpKernel.
 *
 * OpKernelInfo wraps meta data that are used for building OpKernel.
 * Note stateless variables can be hold in OpKernelInfo.
 *
 * OpContext wraps arguments that are passed in Kernel during execution
 * Note stateful variables must be hold in OpKernelContext.
 *
 */

namespace brt {
// this wrapper class is used to create GroupAllocationHook which is applied
// on all arguments of given OpKernelInfo, so the length of the return value of
// \p alloc_f should be the same with the number of the arguments of given
// \p info
class OpKernelGroupAllocationHook : public GroupAllocationHook {
public:
  OpKernelGroupAllocationHook(
      const OpKernelInfo &info,
      std::function<std::vector<AsyncValue>(void)> alloc_f,
      std::function<void(std::vector<AsyncValue>)> free_f)
      : GroupAllocationHook{GetTensorIndexes(info), alloc_f, free_f} {}

private:
  static std::vector<size_t> GetTensorIndexes(const OpKernelInfo &info) {
    size_t nr_args = GetOpArgNum(info);
    std::vector<size_t> tensor_indexes;
    tensor_indexes.reserve(nr_args);
    for (size_t i = 0; i < nr_args; ++i) {
      tensor_indexes.push_back(GetTensorIndexFromOpArgIndex(info, i));
    }
    return tensor_indexes;
  }
};

/**
 * Base class of OpKernel
 */

class OpKernel {
public:
  explicit OpKernel(const OpKernelInfo &info, bool prologue_per_session = false,
                    bool epilogue_per_session = false,
                    bool prologue_per_frame = false,
                    bool epilogue_per_frame = false)
      : info_(info), has_prologue_per_session_(prologue_per_session),
        has_epilogue_per_session_(epilogue_per_session),
        has_prologue_per_frame_(prologue_per_frame),
        has_epilogue_per_frame_(epilogue_per_frame) {}

  virtual ~OpKernel() = default;

  /**
   * ProloguePerSession is called in Session.Preprocess.
   * It is typically used for a one-time preprocess,
   * like layout or some weight binding
   */
  virtual common::Status ProloguePerSession() {
    return common::Status(common::StatusCategory::BRT,
                          common::StatusCode::NOT_IMPLEMENTED,
                          "OpKernel no implemented.");
  };

  /**
   * EpiloguePerSession is called in Session ending.
   */
  virtual common::Status EpiloguePerSession() {
    return common::Status(common::StatusCategory::BRT,
                          common::StatusCode::NOT_IMPLEMENTED,
                          "OpKernel no implemented.");
  };

  /**
   * ProloguePerFrame is called right before every frame.
   * Note. if a frame is reused twice, it is only called once
   * It is typically used for some checking, i/o binding
   */
  virtual common::Status ProloguePerFrame(const ExecutionContext &) {
    return common::Status(common::StatusCategory::BRT,
                          common::StatusCode::NOT_IMPLEMENTED,
                          "OpKernel no implemented.");
  }

  /**
   * EpiloguePerFrame is called right after every frame.
   * Note. if a frame is reused twice, it is only called once
   * It is typically used for some checking, release some resource
   *
   */
  virtual common::Status EpiloguePerFrame(const ExecutionContext &) {
    return common::Status(common::StatusCategory::BRT,
                          common::StatusCode::NOT_IMPLEMENTED,
                          "OpKernel no implemented.");
  }

  /**
   * Compute is called in Session.Run.
   * Note Run only enqueue task into WorkQueue.
   * It may run asynchronously.
   */
  common::Status Run(const ExecutionContext &context) {
    context.event_listener_manager->SignalEvent<Events::BeforeOpKernelRun>(
        {info_});
    auto status = RunImpl(context);
    context.event_listener_manager->SignalEvent<Events::AfterOpKernelRun>(
        {info_});
    return status;
  }
  virtual common::Status RunImpl(const ExecutionContext &) {
    return common::Status(common::StatusCategory::BRT,
                          common::StatusCode::NOT_IMPLEMENTED,
                          "OpKernel no implemented.");
  }

  /**
   * GetGroupAllocationHook used to provide such a hook so that some arguments
   * of this OpKernel could be allocated as a group, see GroupAllocationHook
   * for more details.
   */
  virtual common::Status
  GetGroupAllocationHook(std::unique_ptr<GroupAllocationHook> *) {
    return common::Status::OK();
  }

  const OpKernelInfo &GetOpKernelInfo() const { return info_; }

  bool HasProloguePerSession() const { return has_prologue_per_session_; }
  bool HasEpiloguePerSession() const { return has_epilogue_per_session_; }
  bool HasProloguePerFrame() const { return has_prologue_per_frame_; }
  bool HasEpiloguePerFrame() const { return has_epilogue_per_frame_; }

protected:
  const OpKernelInfo info_; // use value instead of reference here to avoid leak
  bool has_prologue_per_session_;
  bool has_epilogue_per_session_;
  bool has_prologue_per_frame_;
  bool has_epilogue_per_frame_;

private:
  // disallow copy
  OpKernel(const OpKernel &) = delete;

  // disable assignment
  OpKernel &operator=(const OpKernel &) = delete;

  // disallow move
  OpKernel(OpKernel &&) = delete;
  OpKernel &operator=(OpKernel &&) = delete;
};

} // namespace brt
