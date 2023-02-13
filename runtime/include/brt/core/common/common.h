/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Portions Copyright (c) Microsoft Corporation
// Licensed under the MIT License.
// ===========================================================================
// Modification Copyright 2022 ByteDance Ltd. and/or its affiliates.

#pragma once

#include "brt/core/common/code_location.h"
#include "brt/core/common/exceptions.h"
#include "brt/core/common/make_string.h"
#include "brt/core/common/status.h"
#include <algorithm>
#include <chrono>
#include <climits>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace brt {

using TimePoint = std::chrono::high_resolution_clock::time_point;

using common::Status;

#ifdef _WIN32
#define BRT_UNUSED_PARAMETER(x) (x)
#else
#define BRT_UNUSED_PARAMETER(x) (void)(x)
#endif

#ifndef BRT_HAVE_ATTRIBUTE
#ifdef __has_attribute
#define BRT_HAVE_ATTRIBUTE(x) __has_attribute(x)
#else
#define BRT_HAVE_ATTRIBUTE(x) 0
#endif
#endif

// BRT_ATTRIBUTE_UNUSED
//
// Prevents the compiler from complaining about or optimizing away variables
// that appear unused on Linux
#if BRT_HAVE_ATTRIBUTE(unused) || (defined(__GNUC__) && !defined(__clang__))
#undef BRT_ATTRIBUTE_UNUSED
#define BRT_ATTRIBUTE_UNUSED __attribute__((__unused__))
#else
#define BRT_ATTRIBUTE_UNUSED
#endif

#ifdef BRT_NO_EXCEPTIONS
// Print the given final message, the message must be a null terminated char*
// BRT will abort after printing the message.
// For Android, will print to Android system log
// For other platforms, will print to stderr
void PrintFinalMessage(const char *msg);
#endif

// macro to explicitly ignore the return value from a function call so Code
// Analysis doesn't complain
#define BRT_IGNORE_RETURN_VALUE(fn) static_cast<void>(fn)

std::vector<std::string> GetStackTrace();

// these is a helper function that gets defined by platform/Telemetry
void LogRuntimeError(uint32_t session_id, const common::Status &status,
                     const char *file, const char *function, uint32_t line);

// __PRETTY_FUNCTION__ isn't a macro on gcc, so use a check for _MSC_VER
// so we only define it as one for MSVC
#if (_MSC_VER && !defined(__PRETTY_FUNCTION__))
#define __PRETTY_FUNCTION__ __FUNCTION__
#endif

// Capture where a message is coming from. Use __FUNCTION__ rather than the much
// longer __PRETTY_FUNCTION__
#define BRT_WHERE ::brt::CodeLocation(__FILE__, __LINE__, __FUNCTION__)

#define BRT_WHERE_WITH_STACK                                                   \
  ::brt::CodeLocation(__FILE__, __LINE__, __PRETTY_FUNCTION__,                 \
                      ::brt::GetStackTrace())

#ifdef BRT_NO_EXCEPTIONS

#define BRT_TRY if (true)
#define BRT_CATCH(x) else if (false)
#define BRT_RETHROW

// In order to ignore the catch statement when a specific exception (not ... )
// is caught and referred in the body of the catch statements, it is necessary
// to wrap the body of the catch statement into a lambda function. otherwise the
// exception referred will be undefined and cause build break
#define BRT_HANDLE_EXCEPTION(func)

// Throw an exception with optional message.
// NOTE: The arguments get streamed into a string via ostringstream::operator<<
// DO NOT use a printf format string, as that will not work as you expect.
#define BRT_THROW(...)                                                         \
  do {                                                                         \
    ::brt::PrintFinalMessage(                                                  \
        ::brt::BrtException(BRT_WHERE_WITH_STACK,                              \
                            ::brt::MakeString(__VA_ARGS__))                    \
            .what());                                                          \
    abort();                                                                   \
  } while (false)

// Just in order to mark things as not implemented. Do not use in final code.
#define BRT_NOT_IMPLEMENTED(...)                                               \
  do {                                                                         \
    ::brt::PrintFinalMessage(                                                  \
        ::brt::NotImplementedException(::brt::MakeString(__VA_ARGS__))         \
            .what());                                                          \
    abort();                                                                   \
  } while (false)

// Check condition.
// NOTE: The arguments get streamed into a string via ostringstream::operator<<
// DO NOT use a printf format string, as that will not work as you expect.
#define BRT_ENFORCE(condition, ...)                                            \
  do {                                                                         \
    if (!(condition)) {                                                        \
      ::brt::PrintFinalMessage(                                                \
          ::brt::BrtException(BRT_WHERE_WITH_STACK, #condition,                \
                              ::brt::MakeString(__VA_ARGS__))                  \
              .what());                                                        \
      abort();                                                                 \
    }                                                                          \
  } while (false)

#define BRT_THROW_EX(ex, ...)                                                  \
  do {                                                                         \
    ::brt::PrintFinalMessage(                                                  \
        ::brt::MakeString(#ex, "(", ::brt::MakeString(__VA_ARGS__), ")")       \
            .c_str());                                                         \
    abort();                                                                   \
  } while (false)

#else

#define BRT_TRY try
#define BRT_CATCH(x) catch (x)
#define BRT_RETHROW throw;

#define BRT_HANDLE_EXCEPTION(func) func()

// Throw an exception with optional message.
// NOTE: The arguments get streamed into a string via ostringstream::operator<<
// DO NOT use a printf format string, as that will not work as you expect.
#define BRT_THROW(...)                                                         \
  throw ::brt::BrtException(BRT_WHERE_WITH_STACK,                              \
                            ::brt::MakeString(__VA_ARGS__))

// Just in order to mark things as not implemented. Do not use in final code.
#define BRT_NOT_IMPLEMENTED(...)                                               \
  throw ::brt::NotImplementedException(::brt::MakeString(__VA_ARGS__))

// Check condition.
// NOTE: The arguments get streamed into a string via ostringstream::operator<<
// DO NOT use a printf format string, as that will not work as you expect.
#define BRT_ENFORCE(condition, ...)                                            \
  if (!(condition))                                                            \
  throw ::brt::BrtException(BRT_WHERE_WITH_STACK, #condition,                  \
                            ::brt::MakeString(__VA_ARGS__))

#define BRT_THROW_EX(ex, ...) throw ex(__VA_ARGS__)

#endif

#define BRT_MAKE_STATUS(category, code, ...)                                   \
  ::brt::common::Status(::brt::common::category, ::brt::common::code,          \
                        ::brt::MakeString(__VA_ARGS__))

// Check condition. if met, return status.
#define BRT_RETURN_IF(condition, ...)                                          \
  if (condition) {                                                             \
    return ::brt::common::Status(                                              \
        ::brt::common::BRT, ::brt::common::FAIL,                               \
        ::brt::MakeString(BRT_WHERE.ToString(), " ", __VA_ARGS__));            \
  }

// Check condition. if not met, return status.
#define BRT_RETURN_IF_NOT(condition, ...)                                      \
  BRT_RETURN_IF(!(condition), __VA_ARGS__)

// Macros to disable the copy and/or move ctor and assignment methods
// These are usually placed in the private: declarations for a class.

#define BRT_DISALLOW_COPY(TypeName) TypeName(const TypeName &) = delete

#define BRT_DISALLOW_ASSIGNMENT(TypeName)                                      \
  TypeName &operator=(const TypeName &) = delete

#define BRT_DISALLOW_COPY_AND_ASSIGNMENT(TypeName)                             \
  BRT_DISALLOW_COPY(TypeName);                                                 \
  BRT_DISALLOW_ASSIGNMENT(TypeName)

#define BRT_DISALLOW_MOVE(TypeName)                                            \
  TypeName(TypeName &&) = delete;                                              \
  TypeName &operator=(TypeName &&) = delete

#define BRT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TypeName)                        \
  BRT_DISALLOW_COPY_AND_ASSIGNMENT(TypeName);                                  \
  BRT_DISALLOW_MOVE(TypeName)

#define BRT_RETURN_IF_ERROR_SESSIONID(expr, session_id)                        \
  do {                                                                         \
    auto _status = (expr);                                                     \
    if ((!_status.IsOK())) {                                                   \
      ::brt::LogRuntimeError(session_id, _status, __FILE__, __FUNCTION__,      \
                             __LINE__);                                        \
      return _status;                                                          \
    }                                                                          \
  } while (0)

#define BRT_RETURN_IF_ERROR_SESSIONID_(expr)                                   \
  BRT_RETURN_IF_ERROR_SESSIONID(expr, session_id_)
#define BRT_RETURN_IF_ERROR(expr) BRT_RETURN_IF_ERROR_SESSIONID(expr, 0)

#define BRT_THROW_IF_ERROR(expr)                                               \
  do {                                                                         \
    auto _status = (expr);                                                     \
    if ((!_status.IsOK())) {                                                   \
      ::brt::LogRuntimeError(0, _status, __FILE__, __FUNCTION__, __LINE__);    \
      BRT_THROW(_status);                                                      \
    }                                                                          \
  } while (0)

// use this macro when cannot early return
#define CHECK_AND_SET_RETVAL(expr)                                             \
  do {                                                                         \
    if (retval.IsOK()) {                                                       \
      retval = (expr);                                                         \
    }                                                                          \
  } while (0)

// C++ Core Guideline check suppression.
#if defined(_MSC_VER) && !defined(__NVCC__)
#define BRT_GSL_SUPPRESS(tag) [[gsl::suppress(tag)]]
#else
#define BRT_GSL_SUPPRESS(tag)
#endif

inline long long TimeDiffMicroSeconds(TimePoint start_time) {
  auto end_time = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                               start_time)
      .count();
}

inline long long TimeDiffMicroSeconds(TimePoint start_time,
                                      TimePoint end_time) {
  return std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                               start_time)
      .count();
}

struct null_type {};
inline std::string ToMBString(const std::string &s) { return s; }
#ifdef _WIN32
/**
 * Convert a wide character string into a narrow one, with local ANSI code
 * page(like CP936) DO NOT assume the result string is encoded in UTF-8
 */
std::string ToMBString(const std::wstring &s);

std::wstring ToWideString(const std::string &s);
inline std::wstring ToWideString(const std::wstring &s) { return s; }
#else
inline std::string ToWideString(const std::string &s) { return s; }
#endif

#if ((__cplusplus >= 201703L) ||                                               \
     (defined(_MSVC_LANG) && (_MSVC_LANG >= 201703L)))
#define BRT_IF_CONSTEXPR if constexpr
#else
#define BRT_IF_CONSTEXPR if
#endif

} // namespace brt
