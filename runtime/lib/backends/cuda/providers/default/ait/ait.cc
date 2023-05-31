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
#include "brt/core/framework/op_accessor.h"
#include "brt/core/ir/util.h"

#include <cassert>
#include <dlfcn.h>

using namespace brt;
using namespace brt::common;
using namespace brt::cuda;

namespace {

#define LOAD_SYMBOL(var, handle, name_str)                                     \
  var = reinterpret_cast<decltype(var)>(dlsym(handle, name_str));

} // namespace

namespace brt {
namespace cuda {

AITOpKernel::AITOpKernel(const OpKernelInfo &info) : OpKernel(info) {
  OpAccessor accessor(info_);
  std::string ir_path = info_.GetIRPath();
  // get path to bdmodel and load bdmodel
  std::string lib_path = brt::ir::GetParentPath(ir_path);
  lib_path += accessor.GetAttrAsString(std::string("ait_lib_file"));
  aitLibHdl = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  assert(aitLibHdl && "AIT lib .so load failed");

  // load ait funcs
  LOAD_SYMBOL(createFunc_, aitLibHdl, "AITemplateModelContainerCreate");
  LOAD_SYMBOL(deleteFunc_, aitLibHdl, "AITemplateModelContainerDelete");
  LOAD_SYMBOL(runFunc_, aitLibHdl, "AITemplateModelContainerRun");
  LOAD_SYMBOL(getNumInputsFunc_, aitLibHdl,
              "AITemplateModelContainerGetNumInputs");
  LOAD_SYMBOL(getMaximumInputShapeFunc_, aitLibHdl,
              "AITemplateModelContainerGetMaximumInputShape");
  LOAD_SYMBOL(getInputDtypeFunc_, aitLibHdl,
              "AITemplateModelContainerGetInputDtype");
  LOAD_SYMBOL(getNumOutputsFunc_, aitLibHdl,
              "AITemplateModelContainerGetNumOutputs");
  LOAD_SYMBOL(getMaximumOutputShapeFunc_, aitLibHdl,
              "AITemplateModelContainerGetMaximumOutputShape");
  LOAD_SYMBOL(getOutputDtypeFunc_, aitLibHdl,
              "AITemplateModelContainerGetOutputDtype");

  // initialize ait model
  createFunc_(&aitModelHdl, /*num_runtimes*/ 1, nullptr);
  getNumInputsFunc_(aitModelHdl, &numInputs);
  getNumOutputsFunc_(aitModelHdl, &numOutputs);

  // initialize metadata
  inputShapes.reserve(numInputs);
  inputDtypes.reserve(numInputs);
  outputShapes.reserve(numOutputs);
  outputDtypes.reserve(numOutputs);
  for (size_t i = 0; i < numInputs; ++i) {
    AITemplateParamShape shape;
    AITemplateDtype dtype;
    getMaximumInputShapeFunc_(aitModelHdl, i, &shape);
    getInputDtypeFunc_(aitModelHdl, i, &dtype);
    inputShapes.push_back(shape);
    inputDtypes.push_back(dtype);
  }
  aitOutputShapesOut.reserve(numOutputs);
  for (size_t i = 0; i < numOutputs; ++i) {
    AITemplateParamShape shape;
    AITemplateDtype dtype;
    getMaximumOutputShapeFunc_(aitModelHdl, i, &shape);
    getOutputDtypeFunc_(aitModelHdl, i, &dtype);
    outputShapes.push_back(shape);
    outputDtypes.push_back(dtype);
    auto shape_ptr = std::make_unique<int64_t[]>(shape.size);
    aitOutputShapesOut.push_back(shape_ptr.get());
  }
}

AITOpKernel::~AITOpKernel() {
  deleteFunc_(aitModelHdl);
  dlclose(aitLibHdl);
}

common::Status AITOpKernel::RunImpl(const ExecutionContext &ctx) {
  auto work_queue = static_cast<CUDAWorkQueue *>(ctx.work_queue);
  auto cuda_env = work_queue->GetCudaEnv();
  BRT_ENFORCE(cuda_env.IsPrimaryContext(),
              "ait compiler only supports cuda primary context");

  AITData inputs[numInputs], outputs[numOutputs];
  // get inputs
  for (size_t i = 0; i < numInputs; ++i) {
    int tensorId = GetTensorIndexFromOpArgIndex(info_, i);
    void *dataPtr = (void *)(ctx.exec_frame->GetAsyncValueRef(tensorId));
    inputs[i] = AITData(dataPtr, inputShapes[i], inputDtypes[i]);
  }
  // get outputs
  for (size_t i = 0; i < numOutputs; ++i) {
    int tensorId = GetTensorIndexFromOpArgIndex(info_, numInputs + i);
    void *dataPtr = (void *)(ctx.exec_frame->GetAsyncValueRef(tensorId));
    outputs[i] = AITData(dataPtr, outputShapes[i], outputDtypes[i]);
  }

  runFunc_(
      aitModelHdl, inputs, numInputs, outputs, numOutputs,
      reinterpret_cast<AITemplateStreamHandle>(work_queue->GetComputeStream()),
      false /* sync */, false /* graph_mode*/, aitOutputShapesOut.data());
  return Status::OK();
}

} // namespace cuda
} // namespace brt
