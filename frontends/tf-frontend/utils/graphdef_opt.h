//===- graphdef_opt.h -----------------------------------------------------===//
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

#include "tensorflow/core/framework/graph.pb.h"
#include <unordered_map>
#include <vector>

namespace tensorflow {

GraphDef OptimizeGraphDef(const GraphDef &graph_def,
                          const std::vector<std::string> &output_array_vector,
                          const std::vector<std::string> &input_names_vec);

struct InputInfo {
  std::string name;
  std::string dtype;
  std::vector<int64_t> shape;
  std::string shape_str;
};

// this function suggests that the shape of the inputs‘s shape are known except
// the first dimension. if batch_size == -1, then it means batch size is not
// specified.
std::vector<InputInfo> GetInputsByNameWithSpecifiedBatchSize(
    const GraphDef &graphdef, const std::vector<std::string> &names,
    int64_t batch_size, bool force_set_batch_size,
    const std::unordered_map<std::string, std::vector<int>> &name2shape = {});

std::vector<std::string> GetPlaceholderNames(const GraphDef &graphdef);

} // namespace tensorflow