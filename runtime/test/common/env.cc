//===- env.cc -------------------------------------------------*--- C++ -*-===//
//
// Copyright 2022 ByteDance Ltd. and/or its affiliates. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "brt/test/common/env.h"
#include "brt/core/common/logging/sinks/cerr_sink.h"
#include <memory>

namespace brt {
namespace test {

Env *Env::GetInstance() {
  static Env instance;
  return &instance;
}

Env::Env() {
  name_ = "TestEnvLoggingManager";

  logging_manager_ = std::make_unique<brt::logging::LoggingManager>(
      std::unique_ptr<brt::logging::ISink>{
          new brt::logging::CErrSink{}} /*sink*/,
      brt::logging::Severity::kWARNING, false,
      brt::logging::LoggingManager::InstanceType::Default, &name_);
}

} // namespace test
} // namespace brt
