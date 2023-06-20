//===- ait.h --------------------------------------------------*--- C++ -*-===//
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

#include "brt/backends/cuda/providers/default/ait/model_interface.h"
#include "brt/core/framework/op_kernel.h"

namespace brt {
namespace cuda {

struct AITImpl;

class AITOpKernel final : public OpKernel {
public:
  explicit AITOpKernel(const OpKernelInfo &);

  ~AITOpKernel();

  common::Status RunImpl(const ExecutionContext &ctx) override;

private:
  void *aitLibHdl;
  AITemplateModelHandle aitModelHdl;
  size_t numInputs, numOutputs;
  std::vector<AITemplateParamShape> inputShapes, outputShapes;
  std::vector<AITemplateDtype> inputDtypes, outputDtypes;

  decltype(&AITemplateModelContainerCreate) createFunc_ = nullptr;
  decltype(&AITemplateModelContainerDelete) deleteFunc_ = nullptr;
  decltype(&AITemplateModelContainerRun) runFunc_ = nullptr;

  decltype(&AITemplateModelContainerGetNumInputs) getNumInputsFunc_ = nullptr;
  decltype(&AITemplateModelContainerGetMaximumInputShape)
      getMaximumInputShapeFunc_ = nullptr;
  decltype(&AITemplateModelContainerGetInputDtype) getInputDtypeFunc_ = nullptr;

  decltype(&AITemplateModelContainerGetNumOutputs) getNumOutputsFunc_ = nullptr;
  decltype(&AITemplateModelContainerGetMaximumOutputShape)
      getMaximumOutputShapeFunc_ = nullptr;
  decltype(&AITemplateModelContainerGetOutputDtype) getOutputDtypeFunc_ =
      nullptr;
};

} // namespace cuda
} // namespace brt
