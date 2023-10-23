//===- device_api.h -----------------------------------------------*--- C++
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

#include <cstddef>

enum class DeviceType { CPU = 1, CUDA };

struct Device {
  DeviceType device_type_;
  int device_id_;
};

struct DeviceAPI {
  typedef void (*MemcpyH2DFunc)(Device dev, void *dst_ptr, void *src_ptr,
                                size_t nbytes);
  typedef void (*MemcpyH2HFunc)(Device dev, void *dst_ptr, void *src_ptr,
                                size_t nbytes);
  typedef void (*MemcpyD2HFunc)(Device dev, void *dst_ptr, void *src_ptr,
                                size_t nbytes);
  typedef void (*MemcpyD2DFunc)(Device dev, void *dst_ptr, void *src_ptr,
                                size_t nbytes);
  typedef void (*SetDeviceFunc)(Device dev);
  MemcpyH2DFunc MemcpyH2D;
  MemcpyH2HFunc MemcpyH2H;
  MemcpyD2HFunc MemcpyD2H;
  MemcpyD2DFunc MemcpyD2D;
  SetDeviceFunc SetDevice;
};
