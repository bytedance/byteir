
#include "./elementwise_ops.h"
#include "./add.h"
#include "brt/backends/pim/samsung/device/hbm_worker_queue.h"
#include "brt/core/context/execution_context.h"
#include "brt/core/context/execution_frame.h"
#include "brt/core/context/work_queue.h"
#include "brt/core/framework/op_accessor.h"
#include "brt/core/ir/ir.h"
#include "brt/core/ir/util.h"
#include "byteir/Dialect/Byre/ByreDialect.h"

#include <utility>

using namespace brt;
using namespace brt::common;

using namespace brt::ir;
using namespace llvm;
using namespace mlir;
using namespace brt::pim::hbm;
namespace brt {
namespace pim {
namespace hbm {

template <typename T> common::Status Add<T>::ProloguePerSession() {
  std::cout << "this is CustomizeAddOp ProloguePerSession" << std::endl;
  return Status::OK();
}

template <typename T>
common::Status Add<T>::ProloguePerFrame(const ExecutionContext &) {
  std::cout << "this is CustomizeAddOp ProloguePerFrame" << std::endl;
  return Status::OK();
}
template <typename T>
Add<T>::Add(const OpKernelInfo &info)
    : OpKernel(info, true, false, false, false) {
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
 auto work_queue =  static_cast<HBMWorkQueue *>(ctx.work_queue);
  // TODO move the following to util


 

  void *A = accessor.GetArgAsyncValueRef(0);
  void *B = accessor.GetArgAsyncValueRef(1);
  // void *C = accessor.GetResultAsyncValueRef(0);

  auto a_shape = accessor.GetArgShape(0);
  auto b_shape = accessor.GetArgShape(1);
  // auto c_shape = accessor.GetArgShape(2);
  int m = a_shape[0];
  int n = b_shape[1];
  uint32_t output_dim = m * n;
  // auto kernel = static_cast<HBMWorkQueue *>(ctx.work_queue)->Getkenrel();

  // kernel::add_kernel<T>(&kernel, static_cast<T *>(A), static_cast<T *>(B),
  //                       output_dim);

  return common::Status::OK();
};
template class Add<float>;
template class Add<int>;
// template class Add<__half>;
} // namespace hbm
} // namespace pim
} // namespace brt
