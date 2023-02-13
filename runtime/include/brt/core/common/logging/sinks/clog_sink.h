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
/// A std::clog based ISink
/// </summary>
/// <seealso cref="ISink" />
class CLogSink : public OStreamSink {
public:
  CLogSink() : OStreamSink(std::clog, /*flush*/ true) {}
};
} // namespace logging
} // namespace brt
