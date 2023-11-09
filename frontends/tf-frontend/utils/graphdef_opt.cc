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

#include "tensorflow/compiler/jit/shape_inference.h"
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

std::vector<int64_t> GetFinalShapeSizes(
    const std::string &name, const std::vector<int64_t> &shape_sizes_in_graph,
    const std::unordered_map<std::string, std::vector<int>> &name2shape,
    int64_t batch_size, bool force_set_batch_size) {
  std::vector<int64_t> shape_sizes;
  for (size_t j = 0; j < shape_sizes_in_graph.size(); ++j) {
    if (name2shape.count(name) && j < name2shape.at(name).size()) {
      shape_sizes.push_back(name2shape.at(name)[j]);
    } else if (j == 0) {
      if (batch_size > 0) {
        if (shape_sizes_in_graph[j] > 0 && !force_set_batch_size) {
          shape_sizes.push_back(shape_sizes_in_graph[j]);
        } else {
          shape_sizes.push_back(batch_size);
        }
      } else {
        shape_sizes.push_back(shape_sizes_in_graph[j]);
      }
    } else {
      shape_sizes.push_back(shape_sizes_in_graph[j]);
    }
  }
  return shape_sizes;
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

// FIXME: only support feed input into node with one output.
std::vector<InputInfo> tensorflow::GetInputsByNameWithSpecifiedBatchSize(
    const GraphDef &graphdef, const std::vector<std::string> &names,
    int64_t batch_size, bool force_set_batch_size,
    const std::unordered_map<std::string, std::vector<int>> &name2shape) {
  GraphConstructorOptions opts;
  opts.allow_internal_ops = true;
  opts.add_default_attributes = true;
  Graph graph(OpRegistry::Global());
  if (!ConvertGraphDefToGraph(opts, graphdef, &graph).ok()) {
    LOG(ERROR) << "GetInputsByNameWithSpecifiedBatchSize: convert GraphDef to "
                  "Graph failed.";
    assert(0);
  }
  GraphShapeInfo graph_shape_info;
  if (!InferShapes(&graph, /*arg_shapes=*/{}, /*fnlib_def=*/nullptr,
                   &graph_shape_info)
           .ok()) {
    LOG(ERROR) << "GetInputsByNameWithSpecifiedBatchSize: InferShapes failed.";
    assert(0);
  }

  std::vector<InputInfo> res;
  std::unordered_set<std::string> names_set(names.begin(), names.end());
  for (size_t i = 0; i < graphdef.node_size(); ++i) {
    NodeDef node_def = graphdef.node(i);
    // node name
    std::string name = node_def.name();
    if (!names_set.count(name)) {
      continue;
    }

    auto &attr = node_def.attr();
    // shape
    std::vector<int64_t> shape_sizes_in_graph;
    std::string shape_str;
    if (attr.count("shape")) {
      AttrValue shape_attr = node_def.attr().at("shape");
      if (!shape_attr.has_shape()) {
        LOG(ERROR) << node_def.op() << ": " << name
                   << " `shape` attribute isn't type of `TensorShapeProto`. ";
        assert(0);
      }
      const TensorShapeProto &tensor_shape = shape_attr.shape();
      for (size_t j = 0; j < tensor_shape.dim_size(); ++j)
        shape_sizes_in_graph.push_back(tensor_shape.dim(j).size());
    } else {
      auto shape_it = graph_shape_info.find(name);
      if (shape_it == graph_shape_info.end()) {
        LOG(ERROR) << node_def.op() << ": " << name
                   << " shape info can't be fetched from attribute or from "
                      "shape inference.";
        assert(0);
      }
      const std::vector<InferredShape> &inferred_shapes = shape_it->second;
      if (inferred_shapes.size() != 1) {
        LOG(ERROR) << node_def.op() << ": " << name
                   << " more than 1 output is not supported.";
        assert(0);
      }
      for (int64_t dim_size : inferred_shapes[0].shape.dim_sizes())
        shape_sizes_in_graph.push_back(dim_size);
    }
    std::vector<int64_t> shape_sizes =
        GetFinalShapeSizes(name, shape_sizes_in_graph, name2shape, batch_size,
                           force_set_batch_size);
    for (int64_t j = 0; j < shape_sizes.size(); ++j)
      shape_str += (j == 0 ? "" : ",") + std::to_string(shape_sizes[j]);

    // dtype
    AttrValue dtype_attr;
    if (attr.count("dtype")) {
      dtype_attr = node_def.attr().at("dtype");
    } else if (attr.count("output_type")) {
      dtype_attr = node_def.attr().at("output_type");
    } else if (attr.count("T")) {
      dtype_attr = node_def.attr().at("T");
    } else {
      LOG(ERROR) << "Node attribute doesn't contain `dtype` or `T`. ";
      assert(0);
    }
    const DataType dtype = dtype_attr.type();
    std::string dtype_str = DataType_Name(dtype);

    res.push_back({name, dtype_str, shape_sizes, shape_str});
  }
  return res;
}

// note: only return node name, not tensor name
std::vector<std::string>
tensorflow::GetPlaceholderNames(const GraphDef &graphdef) {
  std::vector<std::string> res;
  for (size_t i = 0; i < graphdef.node_size(); ++i) {
    NodeDef node_def = graphdef.node(i);
    if (node_def.op() == "Placeholder") {
      // node name
      std::string name = node_def.name();
      res.push_back(name);
    }
  }
  return res;
}
