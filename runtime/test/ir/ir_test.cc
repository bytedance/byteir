//===- ir_test.cc ---------------------------------------------*--- C++ -*-===//
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

#include "brt/core/common/status.h"
#include "brt/core/ir/ir.h"
#include "brt/test/common/models.h"
#include "brt/test/common/util.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "gtest/gtest.h"
#include <string>

using namespace brt;
using namespace brt::common;
using namespace brt::ir;
using namespace brt::test;
using namespace mlir;

TEST(IRTest, IterateNode) {
  ByREHandle hdl;
  auto status_init = hdl.Initialize();
  BRT_TEST_CHECK_STATUS(status_init);

  ByREBuilder byre_builder;
  auto status_load =
      hdl.LoadFromMemory(CreateAddOp2(byre_builder, "cpu"), "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  Status status_iterate_internal;

  auto status_iterate_final = hdl.IterateNode([&](Operation *op) {
    if (auto byre_op = dyn_cast<byre::ByreOp>(op)) {
      auto key = ByREHandle::GetKey(byre_op);
      if (key != "AddOp_f32f32_f32") {
        status_iterate_internal =
            Status(BRT, FAIL, "Expect get AddOp_f32f32_f32 but get " + key);
        return WalkResult::interrupt();
      }
      for (auto opArg : byre_op->getOperands()) {
        if (opArg.getAsOpaquePointer() == nullptr) {
          status_iterate_internal =
              Status(BRT, FAIL, "IterateNode get a null arg");
          return WalkResult::interrupt();
        }
      }
    }
    return WalkResult::advance();
  });
  BRT_TEST_CHECK_STATUS(status_iterate_final);
  BRT_TEST_CHECK_STATUS(status_iterate_internal);
}

TEST(IRTest, IterateNodeWithInterrupt) {
  ByREHandle hdl;
  auto status_init = hdl.Initialize();
  BRT_TEST_CHECK_STATUS(status_init);

  ByREBuilder byre_builder;
  auto status_load =
      hdl.LoadFromMemory(CreateUnknown(byre_builder, "cpu"), "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  Status status_iterate_internal;

  auto status_iterate_final = hdl.IterateNode([&](Operation *op) {
    if (auto byre_op = dyn_cast<byre::ByreOp>(op)) {
      const std::string key = ByREHandle::GetKey(byre_op);
      if (key == "UnknownOp") {
        status_iterate_internal =
            Status(BRT, FAIL, "IterateNode gets UnknownOp");
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });

  EXPECT_FALSE(status_iterate_final.IsOK());
  EXPECT_FALSE(status_iterate_internal.IsOK());
}

TEST(IRTest, IterateEntryFuncArg) {
  ByREHandle hdl;
  auto status_init = hdl.Initialize();
  BRT_TEST_CHECK_STATUS(status_init);

  ByREBuilder byre_builder;
  auto status_load =
      hdl.LoadFromMemory(CreateAddOp2(byre_builder, "cpu"), "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  Status status_iterate_internal;
  auto status_iterate =
      hdl.IterateEntryFuncArg([&](mlir::BlockArgument block_arg) {
        if (block_arg.getAsOpaquePointer() == nullptr) {
          return Status(BRT, FAIL, "IterateNode get a null arg");
        }
        return Status::OK();
      });
  BRT_TEST_CHECK_STATUS(status_iterate);
}
