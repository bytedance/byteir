// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// ===========================================================================
// Modification Copyright 2022 ByteDance Ltd. and/or its affiliates.

#pragma once

#include "brt/core/common/logging/capture.h"
#include "brt/core/common/logging/isink.h"
#include <ostream>
#include <sstream>
#include <string>

namespace brt {
namespace logging {
/// <summary>
/// A std::ostream based ISink
/// </summary>
/// <seealso cref="ISink" />
class OStreamSink : public ISink {
protected:
  OStreamSink(std::ostream &stream, bool flush)
      : stream_{&stream}, flush_{flush} {}

public:
  void SendImpl(const Timestamp &timestamp, const std::string &logger_id,
                const Capture &message) override;

private:
  std::ostream *stream_;
  const bool flush_;
};
} // namespace logging
} // namespace brt
