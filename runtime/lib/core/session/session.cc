//===- session.cc ---------------------------------------------*--- C++ -*-===//
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

#include "brt/core/session/session.h"

#include "brt/core/context/execution_context.h"
#include "brt/core/context/execution_frame.h"
#include "brt/core/context/work_queue.h"
#include "brt/core/framework/execution_plan.h"
#include "brt/core/framework/execution_provider.h"
#include "brt/core/ir/ir.h"
#include "brt/core/session/request_context.h"
#include <unordered_map>

using namespace brt;
using namespace brt::common;
using namespace brt::ir;

namespace brt {

// Support only ByREHandle now
Session::Session() : ir_handle_(new ByREHandle()) {
  // intialize IR
  ir_handle_->Initialize();
}

Session::~Session() {}

namespace {

inline common::Status CreateAndInitExecutePlan(
    std::unique_ptr<brt::ir::IRHandle> &ir_handle,
    std::unique_ptr<ExecutionPlan> &execution_plan,
    const std::unordered_map<std::string, std::unique_ptr<IAllocator>>
        &allocators,
    const std::vector<std::unique_ptr<ExecutionProvider>> &exec_providers,
    const Device dev, const DeviceAPI *api) {

  ByREHandle &byre_handle = static_cast<ByREHandle &>(*ir_handle);
  execution_plan =
      std::unique_ptr<ExecutionPlan>(new StaticBRTExecutionPlan(byre_handle));

  // TODO decouple this from load??
  auto status_init =
      execution_plan->ProloguePerSession(allocators, exec_providers, dev, api);
  return status_init;
}

} // namespace

common::Status Session::Load(const std::string &url, const std::string &fmt) {

  // load the model
  auto status_load = ir_handle_->Load(url, fmt);

  if (!status_load.IsOK()) {
    return status_load;
  }

  DeviceAPI *device_api = GetDeviceAPI(exec_device_.device_type_);

  return CreateAndInitExecutePlan(ir_handle_, execution_plan_, allocators_,
                                  exec_providers_, exec_device_, device_api);
}

common::Status Session::LoadFromMemory(const void *ptr,
                                       const std::string &fmt) {
  auto status_load = ir_handle_->LoadFromMemory(ptr, fmt);

  if (!status_load.IsOK()) {
    return status_load;
  }

  DeviceAPI *device_api = GetDeviceAPI(exec_device_.device_type_);

  return CreateAndInitExecutePlan(ir_handle_, execution_plan_, allocators_,
                                  exec_providers_, exec_device_, device_api);
}

void *Session::GetWeightAsyncValue(size_t idx) {
  return execution_plan_->GetWeightAsyncValue(idx);
}

common::Status Session::LoadWeights(const std::string &url,
                                    const std::string &fmt) {
  return execution_plan_->LoadWeights(url, fmt);
}

common::Status
Session::NewRequestContext(std::unique_ptr<RequestContext> *request,
                           WorkQueue *work_queue) {
  *request = std::unique_ptr<RequestContext>(new RequestContext(*this));
  // alocate Frame but not allocate Intermediate
  BRT_ENFORCE(execution_plan_ != nullptr);

  execution_plan_->CreateExecutinFrame(&((*request)->frame_));

  if (work_queue) {
    (*request)->SetWorkQueue(work_queue);
  } else {
    execution_plan_->CreateWorkQueue(&((*request)->wq_));
  }

  return Status::OK();
}

common::Status Session::InitializeWeight(size_t id, const void *) {
  BRT_ENFORCE(id < execution_plan_->GetGraphInfo().weight_count);
  return Status::OK();
}

common::Status Session::Run(RequestContext &request) {
  // Create ExecutionContext
  ExecutionContext ctx(request.frame_.get(), request.wq_.get(),
                       execution_plan_->GetFrameStateInfo(),
                       request.events_.get());

  using State = ExecutionFrame::InternalState;
  Status status =
      request.frame_->GetIStateTransition()
          .Edge(State::BeforePrologue, State::MainLoop,
                [&] { return execution_plan_->ProloguePerFrame(ctx); })
          .Invariant(State::MainLoop)
          .Apply();

  if (!status.IsOK()) {
    return status;
  }

  return request.frame_->GetIStateTransition()
      .Edge(State::MainLoop, State::MainLoop,
            [&] { return execution_plan_->Run(ctx); })
      .Apply();
}

void Session::Cleanup(RequestContext &request) noexcept {
  ExecutionContext ctx(request.frame_.get(), request.wq_.get(),
                       execution_plan_->GetFrameStateInfo(),
                       request.events_.get());

  using State = ExecutionFrame::InternalState;
  request.frame_->GetIStateTransition()
      .Edge(State::MainLoop, State::AfterEpilogue,
            [&] { return execution_plan_->EpiloguePerFrame(ctx); })
      // fastpath from BeforePrologue to AfterEpilogue for unused RequestContext
      .Edge(State::BeforePrologue, State::AfterEpilogue)
      .Apply();
}

size_t Session::GetArgNum() {
  return execution_plan_->GetGraphInfo().GetArgNum();
}

size_t Session::GetWeightNum() {
  return execution_plan_->GetGraphInfo().weight_count;
}

const std::vector<std::string> &Session::GetWeightNames() {
  return execution_plan_->GetGraphInfo().weight_names;
}

const std::vector<std::string> &Session::GetInputNames() {
  return execution_plan_->GetGraphInfo().input_names;
}

const std::vector<std::string> &Session::GetOutputNames() {
  return execution_plan_->GetGraphInfo().output_names;
}

const std::vector<std::string> &Session::GetTfInputNamesAttr() {
  return execution_plan_->GetGraphInfo().tf_input_names_attr;
}

const std::vector<std::string> &Session::GetTfOriginalInputNamesAttr() {
  return execution_plan_->GetGraphInfo().tf_original_input_names_attr;
}

const std::vector<std::string> &Session::GetTfOutputNamesAttr() {
  return execution_plan_->GetGraphInfo().tf_output_names_attr;
}

const std::vector<size_t> &Session::GetWeightArgOffsets() {
  return execution_plan_->GetGraphInfo().weight_arg_offsets;
}

const std::vector<size_t> &Session::GetInputArgOffsets() {
  return execution_plan_->GetGraphInfo().input_arg_offsets;
}

const std::vector<size_t> &Session::GetOutputArgOffsets() {
  return execution_plan_->GetGraphInfo().output_arg_offsets;
}

const std::vector<int64_t> Session::GetStaticShape(size_t idx) {
  BRT_ENFORCE(idx < execution_plan_->GetGraphInfo().GetArgNum());
  return execution_plan_->GetStaticShape(idx);
}

DTypeEnum Session::GetDType(size_t idx) {
  BRT_ENFORCE(idx < execution_plan_->GetGraphInfo().GetArgNum());
  return execution_plan_->GetDType(idx);
}

std::string Session::GetSpace(size_t idx) {
  BRT_ENFORCE(idx < execution_plan_->GetGraphInfo().GetArgNum());
  return execution_plan_->GetSpace(idx);
}

int Session::GetGraphArgOffset(const std::string &name) {
  return execution_plan_->GetGraphInfo().GetGraphArgOffset(name);
}

int Session::GetGraphArgAliasOffset(const size_t arg_offset) {
  return execution_plan_->GetGraphInfo().GetGraphArgAliasOffset(arg_offset);
}

common::Status
Session::AddExecutionProvider(std::unique_ptr<ExecutionProvider> provider) {
  exec_providers_.push_back(std::move(provider));
  return common::Status::OK();
}

common::Status Session::AddAllocator(std::unique_ptr<IAllocator> allocator) {
  std::string key = allocator->Info().key;
  if (allocators_.count(key) > 0) {
    return Status(common::StatusCategory::BRT, common::StatusCode::FAIL,
                  "allocator key already exist");
  }
  allocators_.try_emplace(key, std::move(allocator));
  return common::Status::OK();
}

IAllocator *Session::GetAllocator(const std::string &key) {
  if (allocators_.count(key) > 0) {
    return allocators_[key].get();
  }
  return nullptr;
}

DeviceAPI *Session::GetDeviceAPI(const DeviceType &device_type) {
  if (devices_api_.count(device_type)) {
    return devices_api_[device_type];
  }
  return nullptr;
}

common::Status Session::AddDeviceAPI(DeviceType device_type,
                                     DeviceAPI *device_api) {
  devices_api_[device_type] = device_api;
  return common::Status::OK();
}

void Session::SetExecDevice(DeviceType device_type, int device_id) {
  exec_device_ = {device_type, device_id};
}

} // namespace brt
