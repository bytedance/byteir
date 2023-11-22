#include "brt/backends/pim/upmem/providers/default/upmem_provider.h"
#include "brt/backends/common.h"
#include "brt/backends/pim/upmem/device/dpu_allocator.h"
#include "brt/backends/pim/upmem/providers/default/copy/op_registration.h"
#include "brt/backends/pim/upmem/providers/default/gemv/op_registration.h"
#include "brt/backends/pim/upmem/providers/default/math/op_registration.h"
#include "brt/backends/pim/upmem/providers/default/softmax/op_registration.h"
#include "brt/core/framework/kernel_registry.h"
#include "brt/core/session/session.h"
#include <memory>
using namespace brt;
using namespace brt::pim;
using namespace brt::common;

namespace brt {
// namespace pim {
namespace {

// statcially register all UPMEM OpKernels
// TODO: to use MACRO trick to load all kernels
// TODO: to add dynamic suppport.
// clang-format off
BRT_STATIC_KERNEL_REGISTRATION(DeviceKind::UPMEM, ProviderType::BRT, [](KernelRegistry *registry) {
      upmem::RegisterGeMVOp(registry);
      upmem::RegisterSoftmaxOp(registry);
      upmem::RegisterMathOps(registry);
      RegisterCommonBuiltinOps(registry);
    });
// clang-format on

} // namespace

UPMEMExecutionProvider::UPMEMExecutionProvider(const std::string &name)
    : ExecutionProvider(DeviceKind::UPMEM, name) {}

common::Status DefaultUPMEMExecutionProviderFactory(Session *session,
                                                    int /*device_id*/) {
  // create a UPMEM provider
  auto provider = std::make_unique<UPMEMExecutionProvider>();

  // give ownership to the session
  return session->AddExecutionProvider(std::move(provider));
}

} // namespace brt
