#pragma once

#include "brt/backends/common.h"
#include "brt/core/common/status.h"
#include "brt/core/framework/execution_provider.h"

namespace brt {
  class Session;
  // namespace pim{


class HBMPIMExecutionProvider : public ExecutionProvider {
public:
  explicit HBMPIMExecutionProvider(const std::string &name = ProviderType::BRT);
};

// TODO add more option later
common::Status DefaultHBMPIMExecutionProviderFactory(Session *session
                                                   );

// } // namespace brt
}