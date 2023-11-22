#include "brt/backends/pim/samsung/providers/default/hbm_provider.h"
#include "brt/backends/common.h"
#include "brt/backends/pim/samsung/providers/default/add/op_registration.h"
#include "brt/backends/pim/samsung/providers/default/gemv/op_registration.h"
#include "brt/backends/pim/samsung/providers/default/mul/op_registration.h"
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
BRT_STATIC_KERNEL_REGISTRATION(DeviceKind::HBMPIM, ProviderType::BRT, [](KernelRegistry *registry) {

      hbmpim::RegisterAddOp(registry);
      hbmpim::RegisterGEMVOp(registry);
      hbmpim::RegisterMulOp(registry);
      RegisterCommonBuiltinOps(registry);
    });
// clang-format on

} // namespace

HBMPIMExecutionProvider::HBMPIMExecutionProvider(const std::string &name)
    : ExecutionProvider(DeviceKind::HBMPIM, name) {}

common::Status DefaultHBMPIMExecutionProviderFactory(Session *session) {
  // create a HBMPIM provider
  auto provider = std::make_unique<HBMPIMExecutionProvider>();

  // give ownership to the session
  return session->AddExecutionProvider(std::move(provider));
}

} // namespace brt
