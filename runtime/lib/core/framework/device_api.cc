//===- device_api.cc --------------------------------------*--- C++ -*-===//
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

#include "brt/core/framework/device_api.h"

#include <unordered_map>

using namespace brt;
namespace brt {

static std::unordered_map<std::string, DeviceAPI *> &GetDeviceAPIRegistry() {
  static std::unordered_map<std::string, DeviceAPI *> registry;
  return registry;
}

void RegisterDeviceAPI(const std::string &device_name, DeviceAPI *device_api) {
  auto &registry = GetDeviceAPIRegistry();
  registry[device_name] = device_api;
}

const DeviceAPI *GetDeviceAPI(const std::string &device_name) {
  auto &registry = GetDeviceAPIRegistry();
  if (registry.find(device_name) != registry.end()) {
    return registry[device_name];
  }
  return nullptr;
}

DeviceType GetDeviceType(const std::string &name) {
  static std::unordered_map<std::string, DeviceType> str_to_dev_type = {
      {"cpu", DeviceType::CPU}, {"cuda", DeviceType::CUDA}};

  DeviceType dev_type;
  if (str_to_dev_type.count(name) > 0) {
    dev_type = str_to_dev_type[name];
  } else {
    dev_type = DeviceType::UNKNOW;
  }
  return dev_type;
}
} // namespace brt
