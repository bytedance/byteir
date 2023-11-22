#pragma once

#include "brt/backends/common.h"
#include "brt/core/common/status.h"
#include "brt/core/framework/execution_provider.h"
#include <dpu.h>
namespace brt {
class Session;
// namespace pim{

class UPMEMExecutionProvider : public ExecutionProvider {
public:
  explicit UPMEMExecutionProvider(const std::string &name = ProviderType::BRT);
};

// TODO add more option later
common::Status DefaultUPMEMExecutionProviderFactory(Session *session,
                                                    int num_dpus = 1);

// } // namespace brt
} // namespace brt