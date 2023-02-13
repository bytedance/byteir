// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// ===========================================================================
// Modification Copyright 2022 ByteDance Ltd. and/or its affiliates.

#pragma once

namespace brt {
namespace logging {
// mild violation of naming convention. the 'k' lets us use token concatenation
// in the macro
// ::brt::Logging::Severity::k##severity. It's not legal to have
// ::brt::Logging::Severity::##severity the uppercase makes the LOG macro usage
// look as expected for passing an enum value as it will be LOGS(logger, ERROR)
enum class Severity {
  kVERBOSE = 0,
  kINFO = 1,
  kWARNING = 2,
  kERROR = 3,
  kFATAL = 4
};

constexpr const char *SEVERITY_PREFIX = "VIWEF";

} // namespace logging
} // namespace brt
