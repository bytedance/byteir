#pragma once

#include "brt/backends/common.h"
#include "brt/core/common/status.h"
#include "brt/core/framework/execution_provider.h"

namespace brt {
  class Session;
  // namespace pim{


class HBMExecutionProvider : public ExecutionProvider {
public:
  explicit HBMExecutionProvider(const std::string &name = ProviderType::BRT);
};

// TODO add more option later
common::Status DefaultHBMExecutionProviderFactory(Session *session
                                                   );

// } // namespace brt
}