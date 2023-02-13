//===- misc.cc ------------------------------------------------------------===//
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

#include "utils/misc.h"
#include <sstream>

std::string tensorflow::join(const std::vector<std::string> &vec,
                             const std::string &separator) {
  std::stringstream result;
  for (int i = 0; i < vec.size(); ++i) {
    result << vec[i];
    if (i != vec.size() - 1) {
      result << separator;
    }
  }
  return result.str();
}

tensorflow::Status tensorflow::ParseStrToVectorIntMaps(
    absl::string_view name_and_shapes,
    std::unordered_map<std::string, std::vector<int>> &name2shape) {
  std::vector<std::string> name_and_shape_vec =
      absl::StrSplit(name_and_shapes, ':', absl::SkipEmpty());
  for (const auto &name_and_shape : name_and_shape_vec) {
    std::vector<std::string> tokens =
        absl::StrSplit(name_and_shape, ',', absl::SkipEmpty());
    assert(tokens.size() > 1 && "Not valid name and shape str: " &&
           name_and_shape);
    const auto &name = tokens[0];
    std::vector<int> shape;
    for (int i = 1; i < tokens.size(); ++i) {
      shape.push_back(std::stoi(tokens[i]));
    }
    name2shape[name] = shape;
  }
  return tensorflow::OkStatus();
}

tensorflow::Status tensorflow::ParseStrToFloatMaps(
    absl::string_view name_and_value,
    std::unordered_map<std::string, float> &name2value) {
  std::vector<std::string> name_and_value_vec =
      absl::StrSplit(name_and_value, ':', absl::SkipEmpty());
  for (const auto &name_and_value : name_and_value_vec) {
    std::vector<std::string> tokens =
        absl::StrSplit(name_and_value, ',', absl::SkipEmpty());
    assert(tokens.size() == 2 && "Not valid name and value str: " &&
           name_and_value);
    const auto &name = tokens[0];
    std::vector<int> shape;
    name2value[name] = std::stof(tokens[1]);
  }
  return tensorflow::OkStatus();
}
