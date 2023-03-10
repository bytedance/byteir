// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// ===========================================================================
// Modification Copyright 2022 ByteDance Ltd. and/or its affiliates.

#pragma once

#include "brt/core/common/logging/isink.h"
#include "brt/core/common/logging/logging.h"
#include <string>
#include <vector>

namespace brt {
namespace logging {
/// <summary>
/// Class that abstracts multiple ISink instances being written to.
/// </summary>
/// <seealso cref="ISink" />
class CompositeSink : public ISink {
public:
  /// <summary>
  /// Initializes a new instance of the <see cref="CompositeSink"/> class.
  /// Use AddSink to add sinks.
  /// </summary>
  CompositeSink() {}

  /// <summary>
  /// Adds a sink. Takes ownership of the sink (so pass unique_ptr by value).
  /// </summary>
  /// <param name="sink">The sink.</param>
  /// <returns>This instance to allow chaining.</returns>
  CompositeSink &AddSink(std::unique_ptr<ISink> sink) {
    sinks_.push_back(std::move(sink));
    return *this;
  }

private:
  void SendImpl(const Timestamp &timestamp, const std::string &logger_id,
                const Capture &message) override {
    for (auto &sink : sinks_) {
      sink->Send(timestamp, logger_id, message);
    }
  }

  std::vector<std::unique_ptr<ISink>> sinks_;
};
} // namespace logging
} // namespace brt
