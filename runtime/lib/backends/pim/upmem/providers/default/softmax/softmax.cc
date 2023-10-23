// //===- softmax.cc ----------------------------------------------*--- C++
// //-*-===//
// //
// // Copyright 2022 ByteDance Ltd. and/or its affiliates. All rights reserved.
// // Licensed under the Apache License, Version 2.0 (the "License");
// // you may not use this file except in compliance with the License.
// // You may obtain a copy of the License at
// //
// //    http://www.apache.org/licenses/LICENSE-2.0
// //
// // Unless required by applicable law or agreed to in writing, software
// // distributed under the License is distributed on an "AS IS" BASIS,
// // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// // See the License for the specific language governing permissions and
// // limitations under the License.
// //
// //===----------------------------------------------------------------------===//

// #include "./softmax.h"
// #include "brt/backends/pim/upmem/device/dpu.h"
// #include "brt/backends/pim/upmem/device/dpu_call.h"
// #include "brt/backends/pim/upmem/device/upmem_worker_queue.h"

// #include "brt/backends/pim/upmem/device/common.h"
// #include "brt/core/common/utils/math_helper.h"
// #include "brt/core/context/execution_context.h"
// #include "brt/core/context/execution_frame.h"
// #include "brt/core/ir/ir.h"
// #include "brt/core/ir/util.h"

// using namespace brt;
// using namespace brt::common;
// using namespace brt::pim::upmem;
// using namespace brt::ir;

// namespace brt {
// namespace pim {
// namespace upmem {

// SoftmaxOPKernel::SoftmaxOPKernel(const OpKernelInfo &info, int task_type)
//     : OpKernel(info), task_type(task_type) {
//   OpAccessor accessor(info);
//   auto shape_a = accessor.GetArgShape(0);
//   auto shape_b = accessor.GetArgShape(1);
//   auto shape_c = accessor.GetArgShape(2);
//   std::vector<int64_t> dimensions = accessor.GetAttrAsIntArray("dimensions");
//   // A = accessor.GetArgAsyncValueRef(0);
//   // B = accessor.GetArgAsyncValueRef(1);
//   // C = accessor.GetArgAsyncValueRef(2);
// }

// SoftmaxOPKernel::~SoftmaxOPKernel() {}

// common::Status SoftmaxOPKernel::RunImpl(const ExecutionContext &ctx) {
//   auto tensor = GetMLIRValueFromOpArgIndex(info_, 0);
//   auto shape = brt::ir::GetStaticShape(tensor);
//   auto maybeN = LinearizedStaticShape(shape.value());

//   if (!maybeN.has_value()) {
//     return Status(BRT, FAIL, "not supported shape");
//   }
//   int64_t &n = maybeN.value();
//   dpu_info_t dpu_set_info = MakeDPUSet();

//   // TODO move the following to util
//   std::vector<void *> args;
//   args.push_back(&dpu_set_info.dpu_set);    // dpuset
//   args.push_back(&dpu_set_info.dpu);        // dpu
//   args.push_back(&dpu_set_info.nr_of_dpus); // nrdpu
//   auto num_arg = GetOpArgNum(info_);
//   // ptrs is used to make sure args still alive before AddTask is called
//   std::vector<AsyncValueRef> ptrs(num_arg);
//   for (unsigned int i = 0; i < num_arg; ++i) {
//     auto tensor_id = GetTensorIndexFromOpArgIndex(info_, i);
//     ptrs[i] = ctx.exec_frame->GetAsyncValueRef(tensor_id);
//     args.push_back(&ptrs[i]);
//   }
//   // get m and n from shape
//   auto shape = brt::ir::GetStaticShape(tensor);
//   auto maybeN = LinearizedStaticShape(shape.value());
//   if (!maybeN.has_value()) {
//     return Status(BRT, FAIL, "not supported shape");
//   }
//   int64_t &n = maybeN.value();

//   return ctx.work_queue->AddTask(0, nullptr, args.data());
// }
// template class SoftmaxOPKernel<float>;
// template class SoftmaxOPKernel<int>;
// using SoftmaxOPKernel = BRT_UPMEM_CALL(SoftmaxOPKernel(DPU_OK));
// } // namespace upmem
// } // namespace pim
// } // namespace brt
