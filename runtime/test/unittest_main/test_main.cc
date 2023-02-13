//===- test_main.cc -------------------------------------------*--- C++ -*-===//
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

#include "brt/core/common/common.h"
#include "brt/core/framework/execution_provider.h"
#include "brt/test/common/env.h"
#include "gtest/gtest.h"
#include <memory>

#define TEST_MAIN main

using namespace brt;
using namespace brt::test;

int TEST_MAIN(int argc, char **argv) {
  Env *env = Env::GetInstance(); // will creat a singleton
  BRT_UNUSED_PARAMETER(env);

  auto err = brt::ExecutionProvider::StaticRegisterKernelsFromDynlib(
      "lib/libexternal_kernels.so");
  BRT_ENFORCE(err.IsOK(), err.ErrorMessage());

  int status = 0;

  BRT_TRY {
    ::testing::InitGoogleTest(&argc, argv);
    status = RUN_ALL_TESTS();
  }
  BRT_CATCH(const std::exception &ex) {
    BRT_HANDLE_EXCEPTION([&]() {
      std::cerr << ex.what();
      status = -1;
    });
  }
  return status;
}
