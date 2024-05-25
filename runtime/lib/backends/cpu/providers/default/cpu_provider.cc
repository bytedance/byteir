//===- cpu_provider.cc ----------------------------------------*--- C++ -*-===//
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

#include "brt/backends/cpu/providers/default/cpu_provider.h"

#include "./custom_call/repeat.h"
#include "./custom_call/tf_equal.h"
#include "./custom_call/tf_select.h"
#include "./custom_call/tf_string_to_number.h"
#include "./custom_call/tf_where.h"
#include "./custom_call/topk.h"
#include "./llvm/jit.h"
#include "./math/elementwise_ops.h"
#include "./shape/shape_compute.h"
#include "./tensor_generate/fill.h"
#include "./tensor_generate/rng_state.h"
#include "./typecvt/typecvt.h"
#include "brt/backends/common.h"
#include "brt/core/framework/execution_provider.h"
#include "brt/core/session/session.h"
#include "half/half.hpp"

#include <memory>

#define BRT_CPU_DEFAULT_OMP_NUM_THREADS 12

using namespace brt;
using namespace brt::common;

namespace brt {

namespace {

// legacy register
// note: we do not guarantee the backward compatibility of these OpKernels,
// we will remove them if brt upgrade to 2.0
BRT_STATIC_KERNEL_REGISTRATION(
    DeviceKind::CPU, ProviderType::BRT, [](KernelRegistry *registry) {
      registry->Register(
          "AddOp_f32f32_f32",
          [](const brt::OpKernelInfo &info) -> std::shared_ptr<OpKernel> {
            auto kernel = std::shared_ptr<OpKernel>(new cpu::Add<float>(info));
            return kernel;
          });
      registry->Register(
          "Typecvt_i64_i32",
          [](const brt::OpKernelInfo &info) -> std::shared_ptr<OpKernel> {
            auto kernel = std::shared_ptr<OpKernel>(
                new cpu::Typecvt<DTypeEnum::Int64, DTypeEnum::Int32>(info));
            return kernel;
          });
      registry->Register(
          "Typecvt_f32_f16",
          [](const brt::OpKernelInfo &info) -> std::shared_ptr<OpKernel> {
            auto kernel = std::shared_ptr<OpKernel>(
                new cpu::Typecvt<DTypeEnum::Float32, DTypeEnum::Float16>(info));
            return kernel;
          });
      registry->Register(
          "Typecvt_f16_f32",
          [](const brt::OpKernelInfo &info) -> std::shared_ptr<OpKernel> {
            auto kernel = std::shared_ptr<OpKernel>(
                new cpu::Typecvt<DTypeEnum::Float16, DTypeEnum::Float32>(info));
            return kernel;
          });
      registry->Register(
          "tf.Where",
          [](const brt::OpKernelInfo &info) -> std::shared_ptr<OpKernel> {
            return std::make_shared<cpu::TFWhere>(info);
          });
    });

// statcially register all CPU OpKernels
BRT_STATIC_KERNEL_REGISTRATION(
    DeviceKind::CPU, ProviderType::BRT, [](KernelRegistry *registry) {
      registry->Register(
          "LLVMJITOp",
          [](const brt::OpKernelInfo &info) -> std::shared_ptr<OpKernel> {
            return std::make_shared<cpu::LLVMJITOpKernel>(info);
          });
      registry->Register(
          "ComputeShapeOp",
          [](const brt::OpKernelInfo &info) -> std::shared_ptr<OpKernel> {
            return std::make_shared<cpu::ShapeCompute>(info);
          });
      registry->Register(
          "FillOp",
          [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
            return std::make_shared<cpu::Fill>(info);
          });
      registry->Register(
          "tf.Equal",
          [](const brt::OpKernelInfo &info) -> std::shared_ptr<OpKernel> {
            return std::make_shared<cpu::TFEqual>(info);
          });
      registry->Register(
          "byteir.top_k",
          [](const brt::OpKernelInfo &info) -> std::shared_ptr<OpKernel> {
            return std::make_shared<cpu::TopK>(info);
          });
      // registry->Register(
      //     "byteir.repeat",
      //     [](const brt::OpKernelInfo &info) -> std::shared_ptr<OpKernel> {
      //       return std::make_shared<cpu::Repeat>(info);
      //     });
      registry->Register(
          "tf.Select",
          [](const brt::OpKernelInfo &info) -> std::shared_ptr<OpKernel> {
            return std::make_shared<cpu::TFSelect>(info);
          });
      registry->Register(
          "tf.StringToNumber",
          [](const brt::OpKernelInfo &info) -> std::shared_ptr<OpKernel> {
            return std::make_shared<cpu::TFStringToNumber>(info);
          });
      registry->Register(
          "GetSeed",
          [](const brt::OpKernelInfo &info) -> std::shared_ptr<OpKernel> {
            return std::make_shared<cpu::GetSeedOpKernel>(info);
          });
      registry->Register(
          "NextOffset",
          [](const brt::OpKernelInfo &info) -> std::shared_ptr<OpKernel> {
            return std::make_shared<cpu::NextOffsetOpKernel>(info);
          });
      RegisterCommonBuiltinOps(registry);
    });

} // namespace

static CPUExecutionProviderOptions GetDefaultCPUOptions() {
  CPUExecutionProviderOptions options;
  // TODO: fix default value
  options.brt_omp_num_threads = BRT_CPU_DEFAULT_OMP_NUM_THREADS;
  return options;
}

CPUExecutionProvider::CPUExecutionProvider(
    const CPUExecutionProviderOptions &options, const std::string &name)
    : ExecutionProvider(DeviceKind::CPU, name), options_(options) {}

const CPUExecutionProviderOptions &
CPUExecutionProvider::GetProviderOptions() const {
  return options_;
}

common::Status NaiveCPUExecutionProviderFactory(Session *session) {
  // use default CPU provider options
  CPUExecutionProviderOptions default_options = GetDefaultCPUOptions();
  return NaiveCPUExecutionProviderFactory(session, default_options);
}

common::Status
NaiveCPUExecutionProviderFactory(Session *session,
                                 const CPUExecutionProviderOptions &options) {
  // create a CPU provider
  auto cpu_provider = std::make_unique<CPUExecutionProvider>(options);

  // give ownership to the session
  return session->AddExecutionProvider(std::move(cpu_provider));
}

} // namespace brt
