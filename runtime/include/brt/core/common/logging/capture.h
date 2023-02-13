// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// ===========================================================================
// Modification Copyright 2022 ByteDance Ltd. and/or its affiliates.

#pragma once

#include "brt/core/common/code_location.h"
#include "brt/core/common/common.h"
#include "brt/core/common/logging/severity.h"
#include <cstdarg>

namespace brt {
namespace logging {

class Logger;
enum class DataType;

/**
   Class to capture the details of a log message.
*/
class Capture {
public:
  /**
     Initializes a new instance of the Capture class.
     @param logger The logger.
     @param severity The severity.
     @param category The category.
     @param dataType Type of the data.
     @param location The file location the log message is coming from.
  */
  Capture(const Logger &logger, logging::Severity severity,
          const char *category, logging::DataType dataType,
          const CodeLocation &location)
      : logger_{&logger}, severity_{severity}, category_{category},
        data_type_{dataType}, location_{location} {}

  /**
     The stream that can capture the message via operator<<.
     @returns Output stream.
  */
  std::ostream &Stream() noexcept { return stream_; }

#ifdef _MSC_VER
// add SAL annotation for printf format string. requires Code Analysis to run to
// validate usage.
#define BRT_msvc_printf_check _Printf_format_string_
#define __attribute__(x) // Disable for MSVC. Supported by GCC and CLang.
#else
#define BRT_msvc_printf_check
#endif

  /**
     Captures a printf style log message.
     @param name="format">The printf format.
     @param name="">Arguments to the printf format if needed.
     @remarks
     A maximum of 2K of output will be captured currently.
     Non-static method, so 'this' is implicit first arg, and we use
     format(printf(2,3)
  */
  void CapturePrintf(BRT_msvc_printf_check const char *format, ...)
      __attribute__((format(printf, 2, 3)));

  /**
     Process a printf style log message.
     @param format The printf format.
     @param ... Arguments to the printf format if needed.
     @remarks
     A maximum of 2K of output will be captured currently.
     Note: As va_list is 'char *', we have to disambiguate this from
     CapturePrintf so that something like "One string: %s", "the string" does
     not consider "the string" to be the va_list.
  */
  void ProcessPrintf(BRT_msvc_printf_check const char *format, va_list args);

  logging::Severity Severity() const noexcept { return severity_; }

  char SeverityPrefix() const noexcept {
    // Carefully setup so severity_ is a valid index
    return logging::SEVERITY_PREFIX[static_cast<int>(severity_)];
  }

  const char *Category() const noexcept { return category_; }

  logging::DataType DataType() const noexcept { return data_type_; }

  const CodeLocation &Location() const noexcept { return location_; }

  std::string Message() const noexcept { return stream_.str(); }

  ~Capture();

private:
  BRT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Capture);

  const Logger *logger_;
  const logging::Severity severity_;
  const char *category_;
  const logging::DataType data_type_;
  const CodeLocation location_;

  std::ostringstream stream_;
};
} // namespace logging
} // namespace brt
