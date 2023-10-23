//===- cuda_env.h ---------------------------------------------*--- C++ -*-===//
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

#include "dpu_types.h"

#pragma once

namespace brt {
namespace pim {
namespace upmem {
class UpmemEnv {
public:
  UpmemEnv(dpu_set_t dpu_set, dpu_set_t dpu);
  UpmemEnv(uint32_t num_dpus);
  UpmemEnv(const char *dpu_binary_path);

  void Activate();

  const char *GetDpuBinaryPath() { return dpu_binary_path; }
  dpu_set_t *GetDpuSet() { return &dpu_set; }

  int GetNumDpus() { return num_dpus; }
  dpu_set_t GetDpu() { return dpu; }

private:
  void Initialize(dpu_set_t *dpu_set);

  uint32_t num_dpus;
  const char *dpu_binary_path;
  dpu_set_t dpu_set;
  dpu_set_t dpu;
};
} // namespace upmem
} // namespace pim
} // namespace brt