//===- elementwise_ops.cc -------------------------------------*--- C++ -*-===//
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


#include "brt/core/framework/op_accessor.h"
#include "brt/backends/pim/upmem/device/common.h"
#include "brt/core/context/execution_context.h"
#include "brt/core/context/execution_frame.h"
#include "brt/core/context/work_queue.h"
#include "brt/core/ir/ir.h"
#include "brt/core/ir/util.h"
#include "./elementwise_host.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include <dpu.h>
#include <utility>
#include "./elementwise_ops.h"
#ifndef ADD_DPU_BINARY
#define DPU_BINARY "./bin/add_dpu"
#endif
#ifndef SUB_DPU_BINARY
#define DPU_BINARY "./bin/sub_dpu"
#endif
#ifndef MUL_DPU_BINARY
#define DPU_BINARY "./bin/mul_dpu"
#endif
#ifndef DIV_DPU_BINARY
#define DPU_BINARY "./bin/div_dpu"
#endif

using namespace brt;
using namespace brt::common;



using namespace brt::ir;
using namespace llvm;
using namespace mlir;
using namespace brt::pim::upmem;
namespace brt {
namespace pim {
namespace upmem {
template <typename T>
common::Status Add<T>::RunImpl(const ExecutionContext &ctx) {
  OpAccessor accessor(info_, ctx.exec_frame);
 


  // TODO move the following to util
  void *A = accessor.GetArgAsyncValueRef(0);
  void *B = accessor.GetArgAsyncValueRef(1);
  void *C = accessor.GetArgAsyncValueRef(2);


  auto a_shape = accessor.GetArgShape(0);
  auto b_shape = accessor.GetArgShape(1);
  auto c_shape = accessor.GetArgShape(2);
  int m = a_shape[0];
  int n = b_shape[1];



  UpmemEnv env = static_cast<UPMEMWorkQueue *>(ctx.work_queue)->GetUpmemEnv();
  uint32_t nr_of_dpus = MakeDPUSet(env, ADD_DPU_BINARY);

  if (nr_of_dpus
    == 0) { return Status(BRT, FAIL, "no dpus allocated");}

kernel::runadd(env.GetDpuSet(), env.GetDpu(),
                 env.GetNumDpus(), A, B, C, m, n);

  return common::Status::OK();
};
template <typename T>
common::Status Subtract<T>::RunImpl(const ExecutionContext &ctx) {
  OpAccessor accessor(info_, ctx.exec_frame);
 


  // TODO move the following to util
  void *A = accessor.GetArgAsyncValueRef(0);
  void *B = accessor.GetArgAsyncValueRef(1);
  void *C = accessor.GetArgAsyncValueRef(2);

  auto a_shape = accessor.GetArgShape(0);
  auto b_shape = accessor.GetArgShape(1);
  auto c_shape = accessor.GetArgShape(2);
  int m = a_shape[0];
  int n = b_shape[1];



  UpmemEnv env = static_cast<UPMEMWorkQueue *>(ctx.work_queue)->GetUpmemEnv();
  uint32_t nr_of_dpus = MakeDPUSet(env, SUB_DPU_BINARY);

  if (nr_of_dpus
    == 0) { return Status(BRT, FAIL, "no dpus allocated");}

  
kernel::runsub(env.GetDpuSet(), env.GetDpu(),
                 env.GetNumDpus(), A, B, C, m, n);

  return common::Status::OK();
};
template <typename T>
common::Status Mul<T>::RunImpl(const ExecutionContext &ctx) {
  OpAccessor accessor(info_, ctx.exec_frame);
 


  // TODO move the following to util
  void *A = accessor.GetArgAsyncValueRef(0);
  void *B = accessor.GetArgAsyncValueRef(1);
  void *C = accessor.GetArgAsyncValueRef(2);


  auto a_shape = accessor.GetArgShape(0);
  auto b_shape = accessor.GetArgShape(1);
  auto c_shape = accessor.GetArgShape(2);
  int m = a_shape[0];
  int n = b_shape[1];



  UpmemEnv env = static_cast<UPMEMWorkQueue *>(ctx.work_queue)->GetUpmemEnv();
  uint32_t nr_of_dpus = MakeDPUSet(env, MUL_DPU_BINARY);

  if (nr_of_dpus
    == 0) { return Status(BRT, FAIL, "no dpus allocated");}

  
kernel::runmul(env.GetDpuSet(), env.GetDpu(),
                      env.GetNumDpus(), A, B, C, m, n);

  return common::Status::OK();
};
template <typename T>
common::Status Div<T>::RunImpl(const ExecutionContext &ctx) {
  OpAccessor accessor(info_, ctx.exec_frame);
 

  // TODO move the following to util
  void *A = accessor.GetArgAsyncValueRef(0);
  void *B = accessor.GetArgAsyncValueRef(1);
  void *C = accessor.GetArgAsyncValueRef(2);


  auto a_shape = accessor.GetArgShape(0);
  auto b_shape = accessor.GetArgShape(1);
  auto c_shape = accessor.GetArgShape(2);
  int m = a_shape[0];
  int n = b_shape[1];


  UpmemEnv env = static_cast<UPMEMWorkQueue *>(ctx.work_queue)->GetUpmemEnv();
  uint32_t nr_of_dpus = MakeDPUSet(env, DIV_DPU_BINARY);

  if (nr_of_dpus
    == 0) { return Status(BRT, FAIL, "no dpus allocated");}

  
kernel::rundiv(env.GetDpuSet(), env.GetDpu(),
                 env.GetNumDpus(), A, B, C, m, n);

  return common::Status::OK();
};

// instantiate
template class Add<float>;
template class Add<int>;
template class Subtract<float>;
template class Subtract<int>;
template class Mul<float>;
template class Mul<int>;
template class Div<float>;
template class Div<int>;

} // namespace upmem
} // namespace pim
} // namespace brt