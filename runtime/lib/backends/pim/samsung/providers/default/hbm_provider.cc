#include "brt/backends/pim/samsung/providers/default/hbm_provider.h"
#include "brt/backends/common.h"

#include "brt/backends/pim/samsung/providers/default/add/op_registration.h"

#include "brt/core/framework/kernel_registry.h"
#include "brt/core/session/session.h"
#include <memory>
using namespace brt;
using namespace brt::pim;
using namespace brt::common;

namespace brt {
// namespace pim {
namespace {

// statcially register all HBM OpKernels
// TODO: to use MACRO trick to load all kernels
// TODO: to add dynamic suppport.
// clang-format off
BRT_STATIC_KERNEL_REGISTRATION(DeviceKind::HBM, ProviderType::BRT, [](KernelRegistry *registry) {
      hbm::RegisterADDOps(registry);

      RegisterCommonBuiltinOps(registry);
    });
// clang-format on

} // namespace

HBMExecutionProvider::HBMExecutionProvider(const std::string &name)
    : ExecutionProvider(DeviceKind::HBM, name) {}

common::Status DefaultHBMExecutionProviderFactory(Session *session) {
  // create a HBM provider
  auto provider = std::make_unique<HBMExecutionProvider>();

  // give ownership to the session
  return session->AddExecutionProvider(std::move(provider));
}

} // namespace brt
