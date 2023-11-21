//===- dpu_env.h ---------------------------------------------*--- C++ -*-===//
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

#pragma once

#include "brt/core/framework/allocator.h"

namespace brt {
class Session;

class HBMPIMAllocator : public IAllocator {
public:
  HBMPIMAllocator(int device_id, const char *name)
      : IAllocator(BrtMemoryInfo(name, "HBMPIM",
                                 BrtAllocatorType::BrtDeviceAllocator,
                                 device_id, BrtMemTypeDefault)) {}

  void *Alloc(size_t size) override;
  void Free(void *p) override;

private:
  //void CheckDevice(bool throw_when_fail) const;
};
class HBMPIMoutputAllocator : public IAllocator {
public:
  HBMPIMoutputAllocator(int device_id, const char *name)
      : IAllocator(BrtMemoryInfo(name, "HBMPIM_output",
                                 BrtAllocatorType::BrtCustomAllocator,
                                 device_id, BrtMemTypeCPUOutput)) {}

  void *Alloc(size_t size) override;
  void Free(void *p) override;

private:
  // void CheckDevice(bool throw_when_fail) const;
};

class HBMPIMExternalAllocator : public HBMPIMAllocator {
  typedef void *(*ExternalAlloc)(size_t size);
  typedef void (*ExternalFree)(void *p);

public:
  HBMPIMExternalAllocator(int device_id, const char *name, void *alloc,
                          void *free)
      : HBMPIMAllocator(device_id, name) {
    alloc_ = reinterpret_cast<ExternalAlloc>(alloc);
    free_ = reinterpret_cast<ExternalFree>(free);
  }

  void *Alloc(size_t size) override;
  void Free(void *p) override;

private:
  ExternalAlloc alloc_;
  ExternalFree free_;
};

common::Status HBMPIMAllocatorFactory(Session *session, int device_id = 0,
                                      bool use_arena = false,
                                      size_t size = 1 << 30);

} // namespace brt