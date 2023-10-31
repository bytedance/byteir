
#include "./elementwise_ops.h"
#include "./add.h"
#include "brt/core/context/execution_context.h"
#include "brt/core/context/execution_frame.h"
#include "brt/core/context/work_queue.h"
#include "brt/backends/pim/samsung/device/hbm_worker_queue.h"
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
template <typename T>
common::Status Add<T>::RunImpl(const ExecutionContext &ctx) {
  OpAccessor accessor(info_, ctx.exec_frame);

  // TODO move the following to util
  void *A = accessor.GetArgAsyncValueRef(0);
  void *B = accessor.GetArgAsyncValueRef(1);
  // void *C = accessor.GetResultAsyncValueRef(0);


  auto a_shape = accessor.GetArgShape(0);
  auto b_shape = accessor.GetArgShape(1);
  // auto c_shape = accessor.GetArgShape(2);
  int m = a_shape[0];
  int n = b_shape[1];

  auto kernel = static_cast<HBMWorkQueue *>(ctx.work_queue)->Getkenrel();

  kernel::add_kernel(kernel, static_cast<T *>(A), static_cast<T *>(B),
                     m * n);

  return common::Status::OK();
};

} // namespace hbm
} // namespace pim
} // namespace brt  OpAccessor accessor(info_, ctx.exec_frame);
 


