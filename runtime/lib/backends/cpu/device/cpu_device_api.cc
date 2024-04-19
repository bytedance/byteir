//===- cpu_device_api.cc ---------------------------------------------*--- C++
//-*-===//
//
// Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
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

#include "brt/backends/cpu/device/cpu_device_api.h"
#include "brt/core/common/common.h"
#include <cstring>

namespace brt {
namespace {
void SetDevice(Device dev) {}

void MemcpyH2D(Device dev, void *dst_ptr, void *src_ptr, size_t nbytes) {
  BRT_THROW("MemcpyH2D for CPU is not implemented");
}

void MemcpyH2H(Device dev, void *dst_ptr, void *src_ptr, size_t nbytes) {
  memcpy(dst_ptr, src_ptr, nbytes);
}

void MemcpyD2H(Device dev, void *dst_ptr, void *src_ptr, size_t nbytes) {
  BRT_THROW("MemcpyD2H for CPU is not implemented");
}

void MemcpyD2D(Device dev, void *dst_ptr, void *src_ptr, size_t nbytes) {
  BRT_THROW("MemcpyD2D for CPU is not implemented");
}
} // namespace

DeviceAPI *GetCPUDeviceAPI() {
  static DeviceAPI cpu_device_api = {MemcpyH2D, MemcpyH2H, MemcpyD2H, MemcpyD2D,
                                     SetDevice};
  return &cpu_device_api;
}
} // namespace brt
