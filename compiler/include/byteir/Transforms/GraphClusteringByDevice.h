//===- GraphClusteringByDevice.h ------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_TRANSFORMS_GRAPHCLUSTERINGBYDEVICE_H
#define BYTEIR_TRANSFORMS_GRAPHCLUSTERINGBYDEVICE_H

#include "byteir/Transforms/GraphClusteringAlgo.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class ModuleOp;

constexpr StringRef getHostAnchorName() { return "__byteir_host_device__"; }

// Currently the usage of the pass is limited and it may not work correctly in
// non-tensor level dialects. Before this pass, user need to add `device = host`
// attribute to those operations that could only be run on host. Then this pass
// will cluster the host ops and their recursive producers into a host function,
// the other ops will be clustered into a device function.
std::unique_ptr<OperationPass<ModuleOp>> createGraphClusteringByDevicePass(
    std::string attrName = "device", std::string device = "test",
    std::string deviceAnchorName = "__byteir_test_device__",
    bool dupNonSplat = false, bool dupOutputs = false,
    GraphClusteringAlgo clusterAlgo = GraphClusteringAlgo::kFallback);

} // namespace mlir

#endif // BYTEIR_TRANSFORMS_GRAPHCLUSTERINGBYDEVICE_H
