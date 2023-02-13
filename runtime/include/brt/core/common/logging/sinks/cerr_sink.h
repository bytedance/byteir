// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// ===========================================================================
// Modification Copyright 2022 ByteDance Ltd. and/or its affiliates.

#pragma once

#include "brt/core/common/logging/sinks/ostream_sink.h"
#include <iostream>

namespace brt {
namespace logging {
/// <summary>
/// A std::cerr based ISink
/// </summary>
/// <seealso cref="ISink" />
class CErrSink : public OStreamSink {
public:
  CErrSink()
      : OStreamSink(
            std::cerr,
            /*flush*/ false) { // std::cerr isn't buffered so no flush required
  }
};
} // namespace logging
} // namespace brt
