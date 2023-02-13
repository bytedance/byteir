// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// ===========================================================================
// Modification Copyright 2022 ByteDance Ltd. and/or its affiliates.

#include "brt/core/common/logging/capture.h"
#include "brt/core/common/logging/logging.h"

namespace brt {
namespace logging {

void Capture::CapturePrintf(BRT_msvc_printf_check const char *format, ...) {
  va_list arglist;
  va_start(arglist, format);

  ProcessPrintf(format, arglist);

  va_end(arglist);
}

// from https://github.com/KjellKod/g3log/blob/master/src/logcapture.cpp
// LogCapture::capturef License:
// https://github.com/KjellKod/g3log/blob/master/LICENSE Modifications Copyright
// (c) ByteDance.
void Capture::ProcessPrintf(BRT_msvc_printf_check const char *format,
                            va_list args) {
  static constexpr auto kTruncatedWarningText = "[...truncated...]";
  static const int kMaxMessageSize = 2048;
  std::vector<char> message(kMaxMessageSize);

  bool error = false;
  bool truncated = false;

#if (defined(WIN32) || defined(_WIN32) ||                                      \
     defined(__WIN32__) && !defined(__GNUC__))
  errno = 0;
  const int nbrcharacters =
      vsnprintf_s(message.data(), message.size(), _TRUNCATE, format, args);
  if (nbrcharacters < 0) {
    error = errno != 0;
    truncated = !error;
  }
#else
  const int nbrcharacters =
      vsnprintf(message.data(), message.size(), format, args);
  error = nbrcharacters < 0;
  truncated = (nbrcharacters >= 0 &&
               static_cast<size_t>(nbrcharacters) > message.size());
#endif

  if (error) {
    stream_ << "\n\tERROR LOG MSG NOTIFICATION: Failure to successfully parse "
               "the message";
    stream_ << '"' << format << '"' << std::endl;
  } else if (truncated) {
    stream_ << message.data() << kTruncatedWarningText;
  } else {
    stream_ << message.data();
  }
}

Capture::~Capture() {
  if (logger_ != nullptr) {
    logger_->Log(*this);
  }
}
} // namespace logging
} // namespace brt
