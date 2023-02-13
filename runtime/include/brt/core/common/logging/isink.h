// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// ===========================================================================
// Modification Copyright 2022 ByteDance Ltd. and/or its affiliates.

#pragma once

#include "brt/core/common/logging/logging.h"
#include <string>

namespace brt {
namespace logging {
class ISink {
public:
  ISink() = default;

  /**
     Sends the message to the sink.
     @param timestamp The timestamp.
     @param logger_id The logger identifier.
     @param message The captured message.
  */
  void Send(const Timestamp &timestamp, const std::string &logger_id,
            const Capture &message) {
    SendImpl(timestamp, logger_id, message);
  }

  /**
    Sends a Profiling Event Record to the sink.
    @param Profiling Event Record
  */
  virtual void SendProfileEvent(profiling::EventRecord &) const {};

  virtual ~ISink() = default;

private:
  // Make Code Analysis happy by disabling all for now. Enable as needed.
  BRT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ISink);

  virtual void SendImpl(const Timestamp &timestamp,
                        const std::string &logger_id,
                        const Capture &message) = 0;
};
} // namespace logging
} // namespace brt
