//===- ait.cc -------------------------------------------------*--- C++ -*-===//
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

#include "./ait.h"

#include "brt/backends/cuda/device/cuda_work_queue.h"
#include "brt/core/framework/allocator.h"
#include "brt/core/framework/op_accessor.h"
#include "brt/core/ir/util.h"
#include <dlfcn.h>

using namespace brt;
using namespace brt::common;
using namespace brt::cuda;

namespace {

#define LOAD_SYMBOL(var, handle, name_str)                                     \
  var = reinterpret_cast<decltype(var)>(dlsym(handle, name_str));

class AITOpKernelBase {
public:
  AITOpKernelBase(void *aitLibHdl,
                  std::unique_ptr<AITemplateAllocator> allocator)
      : aitLibHdl(aitLibHdl), allocator_(std::move(allocator)) {
    LOAD_SYMBOL(createFunc_, aitLibHdl, "AITemplateModelContainerCreate");
    LOAD_SYMBOL(deleteFunc_, aitLibHdl, "AITemplateModelContainerDelete");
    AIT_ERROR_CHECK(
        createFunc_(&aitModelHdl, /*num_runtimes*/ 1, allocator_.get()));
  }

  ~AITOpKernelBase() noexcept(false) {
    AIT_ERROR_CHECK(deleteFunc_(aitModelHdl));
  }

  AITemplateModelHandle GetModelHdl() { return aitModelHdl; }

protected:
  void *aitLibHdl = nullptr;
  AITemplateModelHandle aitModelHdl;
  std::unique_ptr<AITemplateAllocator> allocator_;

private:
  decltype(&AITemplateModelContainerCreate) createFunc_ = nullptr;
  decltype(&AITemplateModelContainerDelete) deleteFunc_ = nullptr;
};

class AITOpKernelWorkspaceDetector : public AITOpKernelBase {
public:
  class MonotonicTracingOnlyAllocator : public AITemplateAllocator {
  public:
    MonotonicTracingOnlyAllocator(IAllocator *allocator)
        : iallocator_(allocator) {}
    void *Allocate(size_t nbytes) override {
      // Expected no more than 5 times allocation request:
      //  0 for constant
      //  1 for blob
      //  2 for workspace
      //  3 for constant folder's blob
      //  4 for constant folder's workspace
      BRT_ENFORCE(numAllocRequest < 5,
                  "expected no more than 5 times allocation reqeust");
      if (numAllocRequest == 0) {
        // Allocating constants
        void *ptr = iallocator_->Alloc(nbytes);
        numAllocRequest++;
        return ptr;
      }
      if (numAllocRequest == 3 || numAllocRequest == 4) {
        // constant should not be clustered into AIT subgraph, so the constant
        // memory usage and the extra memory used by constant folder should
        // always be zero
        BRT_ENFORCE(nbytes == 0, "expected zero allocation size");
      }

      sizeInBytes += nbytes;
      numAllocRequest++;
      return nullptr;
    }

    void Free(void *ptr) override {
      if (ptr != nullptr) {
        iallocator_->Free(ptr);
      }
    }

    size_t GetTotalAllocatedInBytes() { return sizeInBytes; }

  private:
    size_t sizeInBytes = 0;
    size_t numAllocRequest = 0;
    IAllocator *iallocator_;
  };

  AITOpKernelWorkspaceDetector(void *aitLibHdl, IAllocator *alloc)
      : AITOpKernelBase(
            aitLibHdl, std::make_unique<MonotonicTracingOnlyAllocator>(alloc)) {
  }

  size_t GetWorkspaceSizeInBytes() {
    return static_cast<MonotonicTracingOnlyAllocator *>(allocator_.get())
        ->GetTotalAllocatedInBytes();
  }
};

class AITOpKernelWorkspaceManager {
public:
  void *GetOrAlloc(std::string space, IAllocator *allocator) {
    auto sizeInBytes = space2size[space];
    if (!sizeInBytes)
      return nullptr;

    auto &&iter = space2ptr.find(space);
    if (iter == space2ptr.end()) {
      void *ptr = allocator->Alloc(sizeInBytes);
      space2ptr[space] = std::unique_ptr<void, std::function<void(void *)>>(
          ptr, [=](void *p) { allocator->Free(p); });
      return ptr;
    } else {
      return iter->second.get();
    }
  }

  void Update(std::string space, size_t sizeInBytes) {
    auto &&iter = space2size.find(space);
    if (iter == space2size.end()) {
      space2size[space] = sizeInBytes;
    } else {
      iter->second = std::max(iter->second, sizeInBytes);
    }
  }

private:
  std::unordered_map<std::string, size_t> space2size;
  std::unordered_map<std::string,
                     std::unique_ptr<void, std::function<void(void *)>>>
      space2ptr;
};

class AITOpKernelRunner {
  struct AITOpKernelRunnerImpl : public AITOpKernelBase {
    class MonotonicWorkspaceAllocator : public AITemplateAllocator {
    public:
      MonotonicWorkspaceAllocator(void *workspaceBase, size_t workspaceSize,
                                  IAllocator *allocator)
          : basePtr(workspaceBase), offset(0), sizeInBytes(workspaceSize),
            iallocator_(allocator) {}

      void *Allocate(size_t nbytes) override {
        BRT_ENFORCE(state == 0, "could only allocate in model ctor");
        if (allocConst) {
          allocConst = false;
          if (!nbytes)
            return nullptr;
          void *ptr = iallocator_->Alloc(nbytes);
          constPtr = ptr;
          return ptr;
        }
        if (!nbytes)
          return nullptr;

        void *ptr = static_cast<void *>(static_cast<char *>(basePtr) + offset);
        offset += nbytes;
        BRT_ENFORCE(offset <= sizeInBytes);
        return ptr;
      }

      void Free(void *ptr) override {
        BRT_ENFORCE(state == 2, "could only free in model dtor");
        if (ptr == constPtr) {
          iallocator_->Free(ptr);
        }
      }

      void CallInCtorEpilogue() {
        BRT_ENFORCE(state == 0, "move from ctor to running state");
        state = 1;
      }

      void CallInDtorPrologue() {
        BRT_ENFORCE(state == 1, "move running to dtor state");
        state = 2;
      }

    private:
      int8_t state = 0; // 0 - ctor, 1 - running, 2 - dtor
      void *basePtr;
      size_t offset, sizeInBytes;
      IAllocator *iallocator_;
      bool allocConst = true; // First time alloc is allocating constants
      void *constPtr;
    };

    AITOpKernelRunnerImpl(void *aitLibHdl,
                          AITOpKernelWorkspaceManager *workspaceMgr,
                          IAllocator *alloc, std::string space,
                          size_t workspaceSize, std::string name)
        : AITOpKernelBase(aitLibHdl,
                          std::make_unique<MonotonicWorkspaceAllocator>(
                              workspaceMgr->GetOrAlloc(space, alloc),
                              workspaceSize, alloc)),
          lib_name(name) {
      LOAD_SYMBOL(getNumInputsFunc_, aitLibHdl,
                  "AITemplateModelContainerGetNumInputs");
      LOAD_SYMBOL(getInputNameFunc_, aitLibHdl,
                  "AITemplateModelContainerGetInputName");
      LOAD_SYMBOL(getMaximumInputShapeFunc_, aitLibHdl,
                  "AITemplateModelContainerGetMaximumInputShape");
      LOAD_SYMBOL(getInputDtypeFunc_, aitLibHdl,
                  "AITemplateModelContainerGetInputDtype");
      LOAD_SYMBOL(getNumOutputsFunc_, aitLibHdl,
                  "AITemplateModelContainerGetNumOutputs");
      LOAD_SYMBOL(getOutputNameFunc_, aitLibHdl,
                  "AITemplateModelContainerGetOutputName");
      LOAD_SYMBOL(getMaximumOutputShapeFunc_, aitLibHdl,
                  "AITemplateModelContainerGetMaximumOutputShape");
      LOAD_SYMBOL(getOutputDtypeFunc_, aitLibHdl,
                  "AITemplateModelContainerGetOutputDtype");
      LOAD_SYMBOL(runFunc_, aitLibHdl, "AITemplateModelContainerRun");
      LOAD_SYMBOL(profileFunc_, aitLibHdl, "AITemplateModelContainerProfile");

      AIT_ERROR_CHECK(getNumInputsFunc_(aitModelHdl, &numInputs));
      AIT_ERROR_CHECK(getNumOutputsFunc_(aitModelHdl, &numOutputs));

      inputShapes.reserve(numInputs);
      inputDtypes.reserve(numInputs);
      outputShapes.reserve(numOutputs);
      outputDtypes.reserve(numOutputs);
      for (size_t i = 0; i < numInputs; ++i) {
        const char *name;
        AIT_ERROR_CHECK(getInputNameFunc_(aitModelHdl, i, &name));
        input_name_to_index_.emplace(name, i);
      }
      for (size_t i = 0; i < numOutputs; ++i) {
        const char *name;
        AIT_ERROR_CHECK(getOutputNameFunc_(aitModelHdl, i, &name));
        output_name_to_index_.emplace(name, i);
      }
      for (size_t i = 0; i < numInputs; ++i) {
        AITemplateParamShape shape;
        AITemplateDtype dtype;
        AIT_ERROR_CHECK(getMaximumInputShapeFunc_(aitModelHdl, i, &shape));
        AIT_ERROR_CHECK(getInputDtypeFunc_(aitModelHdl, i, &dtype));
        inputShapes.push_back(shape);
        inputDtypes.push_back(dtype);
      }
      for (size_t i = 0; i < numOutputs; ++i) {
        AITemplateParamShape shape;
        AITemplateDtype dtype;
        AIT_ERROR_CHECK(getMaximumOutputShapeFunc_(aitModelHdl, i, &shape));
        AIT_ERROR_CHECK(getOutputDtypeFunc_(aitModelHdl, i, &dtype));
        outputShapes.push_back(shape);
        outputDtypes.push_back(dtype);
      }

      static_cast<MonotonicWorkspaceAllocator *>(allocator_.get())
          ->CallInCtorEpilogue();
    }

    ~AITOpKernelRunnerImpl() noexcept(false) {
      static_cast<MonotonicWorkspaceAllocator *>(allocator_.get())
          ->CallInDtorPrologue();
    }

    void Run(const std::vector<void *> &args, CUstream_st *stream) {
      AITData inputs[numInputs], outputs[numOutputs];
      for (size_t i = 0; i < numInputs; ++i) {
        std::string input_name = "input_tensor_" + std::to_string(i);
        if (input_name_to_index_.find(input_name) ==
            input_name_to_index_.end()) {
          throw std::runtime_error("input namec" + input_name +
                                   " not found in index map!");
        }
        auto idx = input_name_to_index_.at(input_name);
        inputs[idx] = AITData(args[i], inputShapes[idx], inputDtypes[idx]);
      }
      for (size_t i = 0; i < numOutputs; ++i) {
        std::string output_name = "output_tensor_" + std::to_string(i);
        if (output_name_to_index_.find(output_name) ==
            output_name_to_index_.end()) {
          throw std::runtime_error("output name " + output_name +
                                   " not found in index map!");
        }
        auto idx = output_name_to_index_.at(output_name);
        outputs[idx] =
            AITData(args[numInputs + i], outputShapes[idx], outputDtypes[idx]);
      }

      runFunc_(aitModelHdl, inputs, numInputs, outputs, numOutputs,
               reinterpret_cast<AITemplateStreamHandle>(stream),
               false /* sync */, false /* graph_mode*/,
               nullptr /* output shape */);

      // std::string profile_log = "profile." + lib_name + ".log";
      // const char* filename = profile_log.c_str();
      // profileFunc_(aitModelHdl, inputs, numInputs, outputs, numOutputs,
      //          reinterpret_cast<AITemplateStreamHandle>(stream),
      //          5 /* num_iters */, filename /* filename */);
    }

    size_t numInputs, numOutputs;
    std::vector<AITemplateParamShape> inputShapes, outputShapes;
    std::vector<AITemplateDtype> inputDtypes, outputDtypes;
    std::unordered_map<std::string, size_t> input_name_to_index_;
    std::unordered_map<std::string, size_t> output_name_to_index_;
    std::string lib_name;

    decltype(&AITemplateModelContainerRun) runFunc_ = nullptr;
    decltype(&AITemplateModelContainerProfile) profileFunc_ = nullptr;
    decltype(&AITemplateModelContainerGetInputName) getInputNameFunc_ = nullptr;
    decltype(&AITemplateModelContainerGetNumInputs) getNumInputsFunc_ = nullptr;
    decltype(&AITemplateModelContainerGetMaximumInputShape)
        getMaximumInputShapeFunc_ = nullptr;
    decltype(&AITemplateModelContainerGetInputDtype) getInputDtypeFunc_ =
        nullptr;

    decltype(&AITemplateModelContainerGetOutputName) getOutputNameFunc_ =
        nullptr;
    decltype(&AITemplateModelContainerGetNumOutputs) getNumOutputsFunc_ =
        nullptr;
    decltype(&AITemplateModelContainerGetMaximumOutputShape)
        getMaximumOutputShapeFunc_ = nullptr;
    decltype(&AITemplateModelContainerGetOutputDtype) getOutputDtypeFunc_ =
        nullptr;
  };

public:
  template <typename... Args>
  AITOpKernelRunner(Args &&...args)
      : runnerImpl(nullptr), ctors{std::forward<Args>(args)...} {}

  template <typename... Args> void Run(Args &&...args) {
    if (!runnerImpl) {
      runnerImpl.reset(new AITOpKernelRunnerImpl(
          std::make_from_tuple<AITOpKernelRunnerImpl>(ctors)));
    }
    runnerImpl->Run(std::forward<Args>(args)...);
  }

private:
  std::unique_ptr<AITOpKernelRunnerImpl> runnerImpl;
  std::tuple<void *, AITOpKernelWorkspaceManager *, IAllocator *, std::string,
             size_t, std::string>
      ctors;
};
} // namespace

namespace brt {
namespace cuda {

AITOpKernel::AITOpKernel(const OpKernelInfo &info)
    : OpKernel(info, false, false, true, true) {
  OpAccessor accessor(info_);
  std::string ir_path = info_.GetIRPath();
  // get path to bdmodel and load bdmodel
  std::string lib_path = brt::ir::GetParentPath(ir_path);
  lib_path += accessor.GetAttrAsString(std::string("ait_lib_file"));
  aitLibHdl = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  BRT_ENFORCE(aitLibHdl != nullptr, "AIT lib .so load failed");
  std::string space = accessor.GetAttrAsString("device");
  IAllocator *alloc = info_.GetAllocator(space);
  workspaceSizeInBytes =
      AITOpKernelWorkspaceDetector(aitLibHdl, alloc).GetWorkspaceSizeInBytes();
}

AITOpKernel::~AITOpKernel() { dlclose(aitLibHdl); }

common::Status AITOpKernel::RunImpl(const ExecutionContext &ctx) {
  OpAccessor accessor(info_, ctx.exec_frame);
  auto work_queue = static_cast<CUDAWorkQueue *>(ctx.work_queue);
  auto cuda_env = work_queue->GetCudaEnv();
  BRT_ENFORCE(cuda_env.IsPrimaryContext(),
              "ait compiler only supports cuda primary context");

  std::vector<AsyncValue> args;
  for (size_t i = 0; i < accessor.GetNumArgs(); ++i) {
    args.push_back(accessor.GetArgAsyncValueRef(i));
  }

  brt::ExecutionFrame::StateInfo &state_info = ctx.frame_state_info;
  size_t handle_offset =
      state_info.GetStateOffset(GetAITOpKernelRunnerUniqueKey());
  AITOpKernelRunner *runner = reinterpret_cast<AITOpKernelRunner *>(
      ctx.exec_frame->GetState(handle_offset));

  runner->Run(args, work_queue->GetComputeStream());

  return Status::OK();
}

common::Status AITOpKernel::ProloguePerFrame(const ExecutionContext &ctx) {
  brt::ExecutionFrame::StateInfo &state_info = ctx.frame_state_info;
  auto status = state_info.CreateStateIfNotExist(
      GetAITOpKernelWorkspaceManagerSharedKey(), ctx.exec_frame,
      []() { return static_cast<void *>(new AITOpKernelWorkspaceManager()); });

  if (!status.IsOK())
    return status;

  size_t handle_offset =
      state_info.GetStateOffset(GetAITOpKernelWorkspaceManagerSharedKey());
  AITOpKernelWorkspaceManager *workspaceMgr =
      reinterpret_cast<AITOpKernelWorkspaceManager *>(
          ctx.exec_frame->GetState(handle_offset));
  std::string space = OpAccessor(info_).GetAttrAsString("device");
  std::string name =
      OpAccessor(info_).GetAttrAsString(std::string("ait_lib_file"));
  workspaceMgr->Update(space, workspaceSizeInBytes);

  IAllocator *alloc = info_.GetAllocator(space);
  if (!alloc)
    return common::Status(common::StatusCategory::BRT, common::StatusCode::FAIL,
                          "Cannot find allocator");

  status = state_info.CreateStateIfNotExist(
      GetAITOpKernelRunnerUniqueKey(), ctx.exec_frame, [=]() {
        return static_cast<void *>(new AITOpKernelRunner(
            aitLibHdl, workspaceMgr, alloc, space, workspaceSizeInBytes, name));
      });
  return status;
}

common::Status AITOpKernel::EpiloguePerFrame(const ExecutionContext &ctx) {
  brt::ExecutionFrame::StateInfo &state_info = ctx.frame_state_info;
  {
    size_t offset =
        state_info.GetStateOffset(GetAITOpKernelWorkspaceManagerSharedKey());
    void *ptr = ctx.exec_frame->GetAndResetState(offset);
    if (ptr != nullptr) {
      AITOpKernelWorkspaceManager *mgr =
          static_cast<AITOpKernelWorkspaceManager *>(ptr);
      delete mgr;
    }
  }
  {
    size_t offset = state_info.GetStateOffset(GetAITOpKernelRunnerUniqueKey());
    void *ptr = ctx.exec_frame->GetAndResetState(offset);
    if (ptr != nullptr) {
      AITOpKernelRunner *runner = static_cast<AITOpKernelRunner *>(ptr);
      delete runner;
    }
  }
  return brt::common::Status::OK();
}

std::string AITOpKernel::GetAITOpKernelWorkspaceManagerSharedKey() {
  // shared between all instances of the ait op kernel
  return "ait_op_kernel_workspace";
}

std::string AITOpKernel::GetAITOpKernelRunnerUniqueKey() {
  return "ait_op_kernel_runner" + OpAccessor(info_).GetUID();
}

} // namespace cuda
} // namespace brt
