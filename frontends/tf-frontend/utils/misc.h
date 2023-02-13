//===- misc.h -------------------------------------------------------------===//
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

#ifndef TFEXT_UTILS_MISC_H_
#define TFEXT_UTILS_MISC_H_

#include <string>
#include <vector>

#include "absl/strings/str_split.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {

std::string join(const std::vector<std::string> &vec,
                 const std::string &separator);

tensorflow::Status ParseStrToVectorIntMaps(
    absl::string_view name_and_shapes,
    std::unordered_map<std::string, std::vector<int>> &name2shape);

tensorflow::Status
ParseStrToFloatMaps(absl::string_view name_and_value,
                    std::unordered_map<std::string, float> &name2value);

} // namespace tensorflow

#endif // TFEXT_UTILS_MISC_H_