//===- cuda_device_api.cc ---------------------------------------------*--- C++
//-*-===//
//
// Copyright 2023 ByteDance Ltd. and/or its affiliates. All rights reserved.
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

#include "brt/backends/cuda/device/cuda_device_api.h"
#include "brt/backends/cuda/device/common/cuda_call.h"
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>

namespace brt {
namespace {
void SetDevice(Device dev) { BRT_CUDA_CALL(cudaSetDevice(dev.device_id_)); }

void MemcpyH2D(Device dev, void *dst_ptr, void *src_ptr, size_t nbytes) {
  SetDevice(dev);
  BRT_CUDA_CALL(cudaMemcpy(dst_ptr, src_ptr, nbytes, cudaMemcpyHostToDevice));
}

void MemcpyH2H(Device dev, void *dst_ptr, void *src_ptr, size_t nbytes) {
  memcpy(dst_ptr, src_ptr, nbytes);
}

void MemcpyD2H(Device dev, void *dst_ptr, void *src_ptr, size_t nbytes) {
  SetDevice(dev);
  BRT_CUDA_CALL(cudaMemcpy(dst_ptr, src_ptr, nbytes, cudaMemcpyDeviceToHost));
}

void MemcpyD2D(Device dev, void *dst_ptr, void *src_ptr, size_t nbytes) {
  SetDevice(dev);
  BRT_CUDA_CALL(cudaMemcpy(dst_ptr, src_ptr, nbytes, cudaMemcpyDeviceToDevice));
}
} // namespace

DeviceAPI *GetCUDADeviceAPI() {
  static DeviceAPI cuda_device_api = {MemcpyH2D, MemcpyH2H, MemcpyD2H,
                                      MemcpyD2D, SetDevice};
  return &cuda_device_api;
}
} // namespace brt
