//===- graphdef_opt.cc ----------------------------------------------------===//
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

#include "utils/graphdef_opt.h"

#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/grappler/clusters/single_machine.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/meta_optimizer.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/tools/graph_transforms/transform_graph.h"

#include <cassert>
#include <unordered_set>

using namespace tensorflow;
using namespace tensorflow::grappler;
using namespace tensorflow::graph_transforms;

namespace {

// Note the graphdef should already be optimized by DCE
std::vector<std::string>
GetNeededInputs(const GraphDef &graphdef,
                const std::vector<std::string> &input_names_vec) {
  GraphConstructorOptions options;
  options.allow_internal_ops = true;
  Graph graph(OpRegistry::Global());
  Status s = ConvertGraphDefToGraph(options, std::move(graphdef), &graph);
  if (!s.ok()) {
    LOG(ERROR) << "convert graph def to graph failed.";
  }

  std::unordered_set<std::string> used;
  for (auto iter : graph.op_nodes()) {
    for (auto inp_iter : iter->in_nodes()) {
      used.insert(inp_iter->name());
    }
  }
  std::vector<std::string> res;
  for (const std::string &name : input_names_vec) {
    if (used.count(name)) {
      res.push_back(name);
    }
  }
  return res;
}

GraphDef
OptimizeGraphDefInternal(const GraphDef &graph_def,
                         const std::vector<std::string> &output_array_vector,
                         const std::vector<std::string> &input_names_vec) {
  GrapplerItem item;
  item.id = "tf_graph";
  item.graph = graph_def;
  for (const std::string &str : output_array_vector) {
    item.fetch.emplace_back(NodeName(str));
  }
  for (const std::string &str : input_names_vec) {
    item.feed.emplace_back(NodeName(str), Tensor());
  }

  ConfigProto config_proto;
  RewriterConfig *rewriter_config =
      config_proto.mutable_graph_options()->mutable_rewrite_options();
  rewriter_config->set_remapping(RewriterConfig::OFF);
  MetaOptimizer optimizer(nullptr, config_proto);
  GraphDef output;
  SingleMachine cluster(10, 1, 0);
  const Status status = optimizer.Optimize(&cluster, item, &output);
  assert(status.ok());
  return output;
}
} // namespace

GraphDef tensorflow::OptimizeGraphDef(
    const GraphDef &graph_def,
    const std::vector<std::string> &output_array_vector,
    const std::vector<std::string> &input_names_vec) {
  GraphDef gdef =
      OptimizeGraphDefInternal(graph_def, output_array_vector, input_names_vec);
  std::vector<std::string> needed_inputs =
      GetNeededInputs(gdef, input_names_vec);
  GraphDef final_gdef =
      OptimizeGraphDefInternal(gdef, output_array_vector, needed_inputs);
  return final_gdef;
}

std::vector<InputInfo> tensorflow::GetPlaceholderInputsWithSpecifiedBatchSize(
    const GraphDef &graphdef, int64_t batch_size, bool force_set_batch_size,
    const std::unordered_map<std::string, std::vector<int>> &name2shape) {
  std::vector<InputInfo> res;
  for (int i = 0; i < graphdef.node_size(); ++i) {
    NodeDef node_def = graphdef.node(i);
    if (node_def.op() == "Placeholder") {
      // name
      std::string name = node_def.name();
      auto &attr = node_def.attr();

      // shape
      if (!attr.count("shape")) {
        LOG(ERROR) << "Placeholder attribute doesn't contain `shape`. ";
        assert(0);
      }
      AttrValue shape_attr = node_def.attr().at("shape");
      if (!shape_attr.has_shape()) {
        LOG(ERROR) << "Placeholder `shape` attribute isn't type of "
                      "`TensorShapeProto`. ";
        assert(0);
      }
      const TensorShapeProto &tensor_shape = shape_attr.shape();
      std::vector<int64_t> shape_sizes;
      std::string shape_str;
      for (int j = 0; j < tensor_shape.dim_size(); ++j) {
        if (name2shape.count(name) && j < name2shape.at(name).size()) {
          shape_sizes.push_back(name2shape.at(name)[j]);
        } else if (j == 0) {
          if (batch_size > 0) {
            if (tensor_shape.dim(j).size() > 0 && !force_set_batch_size) {
              shape_sizes.push_back(tensor_shape.dim(j).size());
            } else {
              shape_sizes.push_back(batch_size);
            }
          } else {
            shape_sizes.push_back(tensor_shape.dim(j).size());
          }
        } else {
          shape_sizes.push_back(tensor_shape.dim(j).size());
        }

        shape_str += (j == 0 ? "" : ",") + std::to_string(shape_sizes[j]);
      }

      // dtype
      if (!attr.count("dtype")) {
        LOG(ERROR) << "Placeholder attribute doesn't contain `dtype`. ";
        assert(0);
      }
      AttrValue dtype_attr = node_def.attr().at("dtype");
      const DataType dtype = dtype_attr.type();
      std::string dtype_str = DataType_Name(dtype);

      res.push_back({name, dtype_str, shape_sizes, shape_str});
    }
  }
  return res;
}