
#include "./elementwise_ops.h"
#include "./add.h"
#include "FP16.h"
#include "brt/backends/pim/samsung/device/BurstTensor.h"
#include "brt/backends/pim/samsung/device/hbm_worker_queue.h"
#include "brt/core/context/execution_context.h"
#include "brt/core/context/execution_frame.h"
#include "brt/core/context/work_queue.h"
#include "brt/core/framework/op_accessor.h"
#include "brt/core/ir/ir.h"
#include "brt/core/ir/util.h"

#include <utility>

using namespace brt;
using namespace brt::common;

using namespace brt::ir;
using namespace llvm;
using namespace mlir;
using namespace brt::pim::hbmpim;
namespace brt {
namespace pim {
namespace hbmpim {

template <typename T> common::Status Add<T>::ProloguePerSession() {
  std::cout << "this is CustomizeAddOp ProloguePerSession" << std::endl;
  return Status::OK();
}

template <typename T>
common::Status Add<T>::ProloguePerFrame(const ExecutionContext &ctx) {
  brt::ExecutionFrame::StateInfo &state_info = ctx.frame_state_info;

  std::string space = OpAccessor(info_).GetAttrAsString("device");

  IAllocator *alloc = info_.GetAllocator(space);
  if (!alloc)
    return common::Status(common::StatusCategory::BRT, common::StatusCode::FAIL,
                          "Cannot find allocator");

  // status = state_info.CreateStateIfNotExist(
  //     GetAITOpKernelRunnerUniqueKey(), ctx.exec_frame, [=]() {
  //       return static_cast<void *>(new AITOpKernelRunner(
  //           aitLibHdl, workspaceMgr, alloc, space, workspaceSizeInBytes,
  //           name));
  //     });
}
template <typename T> Add<T>::Add(const OpKernelInfo &info) : OpKernel(info) {
  OpAccessor accessor(info_);
  std::string ir_path = info_.GetIRPath();
  // std::string lib_path = brt::ir::GetParentPath(ir_path);
  //  lib_path += accessor.GetAttrAsString(std::string("hbm_lib_file"));
  std::string space = accessor.GetAttrAsString("device");
  // IAllocator *alloc = info_.GetAllocator(space);
}
template <typename T>
common::Status Add<T>::RunImpl(const ExecutionContext &ctx) {
  OpAccessor accessor(info_, ctx.exec_frame);

  auto work_queue = ctx.work_queue;
  if (work_queue == nullptr) {
    return common::Status(common::StatusCategory::BRT, common::StatusCode::FAIL,
                          "Work queue is none");
  }
  // TODO move the following to util

  auto A = GetMLIRValueFromOpArgIndex(info_, 0);
  auto B = GetMLIRValueFromOpArgIndex(info_, 1);
  auto C = GetMLIRValueFromOpArgIndex(info_, 2);
  auto shape = brt::ir::GetStaticShape(A);
  auto maybeN = LinearizedStaticShape(shape.value());
  if (!maybeN.has_value()) {
    return Status(BRT, FAIL, "not supported shape");
  }
  auto &n = maybeN.value();

  //   auto tensor_id_A = GetTensorIndexFromOpArgIndex(info_, 0);
  // AsyncValueRef  a_ptr = ctx.exec_frame->GetAsyncValueRef(tensor_id_A);
  //   auto tensor_id_B = GetTensorIndexFromOpArgIndex(info_, 1);
  //  AsyncValueRef b_ptr = ctx.exec_frame->GetAsyncValueRef(tensor_id_A);

  // T *A = static_cast<T *>(accessor.GetArgAsyncValueRef(0));
  // T *B = static_cast<T *>(accessor.GetArgAsyncValueRef(1));

  // // T *C = static_cast<T *>(accessor.Get(0));
  // T *C = static_cast<T *>(accessor.GetArgAsyncValueRef(2));

  // auto a_shape = accessor.GetArgShape(0);
  // auto b_shape = accessor.GetArgShape(1);
  // auto c_shape = accessor.GetArgShape(2);
  // auto c_shape = accessor.GetArgShape(2);
  // int m = a_shape[0];
  // int n = b_shape[1];

  uint32_t output_dim = static_cast<uint32_t>(n);
  vector<float> c = vector<float>(n);
  vector<float> d = vector<float>(n);
  vector<float> x = vector<float>(n);

  auto kernel = *static_cast<HBMPIMWorkQueue *>(ctx.work_queue)->Getkernel();
  // auto kernel = ->Getkernel();
  // // BurstTensor<T> C(output_dim);

  TDataDim *dim_data = new TDataDim(KernelType::ADD, 1, n, n, true, c, d, x);

  kernel.preloadNoReplacement(&dim_data->input_npbst_, 0, 0);
  kernel.preloadNoReplacement(&dim_data->input1_npbst_, 0, 0);

  DRAMSim::BurstType *r = new DRAMSim::BurstType[n];
  // TODO move the following to util
  std::vector<void *> args;
  // args.push_back(kernel);         // grid
  // args.push_back(&dim_data);        // block
  // args.push_back(&r); // dyn_shared_size

  // auto num_arg = GetOpArgNum(info_);
  // ptrs is used to make sure args still alive before AddTask is called

  // std::vector<AsyncValueRef> ptrs(num_arg);
  // for (unsigned int i = 0; i < num_arg; ++i) {
  //   auto tensor_id = GetTensorIndexFromOpArgIndex(info_, i);
  //   ptrs[i] = ctx.exec_frame->GetAsyncValueRef(tensor_id);
  //   args.push_back(&ptrs[i]);
  // }

  // args.push_back(&n);
  // // return common::Status(common::StatusCategory::BRT,
  // common::StatusCode::FAIL,
  // //                       "Cannot c allocator");
  kernel::add_kernel<T>(kernel, dim_data, r);

  // work_queue->AddTask(0, NULL, NULL);

  // return common::Status(common::StatusCategory::BRT,
  // common::StatusCode::FAIL, "Cannot cc allocator");
  // return static_cast<HBMPIMWorkQueue*>(ctx.work_queue)->AddTask(0, (void
  // *)kernel::add_kernel<T>, args.data());
  return common::Status::OK();
};
template class Add<float>;
template class Add<int>;
template class Add<half_float::half>;
} // namespace hbmpim
} // namespace pim
} // namespace brt