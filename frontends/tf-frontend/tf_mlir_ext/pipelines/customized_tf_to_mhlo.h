//===- customized_tf_to_mhlo.h --------------------------------------------===//
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

#ifndef TFEXT_PIPELINES_CUSTOMIZED_TF_TO_MHLO_H
#define TFEXT_PIPELINES_CUSTOMIZED_TF_TO_MHLO_H

#include "mlir/Pass/Pass.h"
#include <memory>
#include <string>
#include <vector>

namespace mlir {
class ModuleOp;

namespace tfext {

std::unique_ptr<OperationPass<ModuleOp>> createCustomizedTfToMhloPipelinePass(
    const std::vector<std::string> &customcall_ops = {},
    bool remove_control_flow = false, bool staticalize_dynamic_shape = false,
    bool stop_after_convert_to_tf_dialect = false,
    bool stop_after_rewrite_custom_call = false,
    const std::unordered_map<std::string, Attribute>
        &additional_main_func_attrs = {},
    bool set_assuming_to_be_true = true, int64_t repeat_out_batch_size = -1);

} // namespace tfext
} // namespace mlir

#endif // TFEXT_PIPELINES_CUSTOMIZED_TF_TO_MHLO_H
