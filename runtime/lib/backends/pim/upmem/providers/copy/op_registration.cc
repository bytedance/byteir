#include "brt/backends/cuda/providers/default/copy/op_registration.h"

#include "./copy.h"
#include "brt/core/framework/kernel_registry.h"

namespace brt {
namespace pim {

namespace upmem {
void RegisterCopyOps(KernelRegistry *registry) {
  registry->Register(
      "preparexfr",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        return std::make_shared<PrepareXfrOpKernel>(info, 1);
      });

  registry->Register(
      "pushxfr",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        return std::make_shared<PushXfrOpKernel>(info, 2);
      });
}
} // namespace upmem
} // namespace pim
} // namespace brt
