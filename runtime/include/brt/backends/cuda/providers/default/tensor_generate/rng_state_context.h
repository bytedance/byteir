//===- rng_state_context.h --------------------------------------*--- C++
//-*-===//
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

#pragma once

namespace brt {
namespace cuda {

class rngStateContext {
private:
  int64_t seed;
  int64_t offset;

public:
  explicit rngStateContext() : seed(0), offset(0) {}

  int64_t getSeed() { return seed; }

  int64_t nextOffset() { return offset++; }

  void setSeed(int64_t seed_) { seed = seed_; }
};

using rngStateHandle_t = rngStateContext *;

} // namespace cuda
} // namespace brt