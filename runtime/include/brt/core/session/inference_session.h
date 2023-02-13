//===- inference_session.h ------------------------------------*--- C++ -*-===//
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

#pragma once

#include "brt/core/session/session.h"
#include <string>
#include <unordered_map>

namespace brt {

// forwarding
class ExecutionProvider

    class InferenceSession : public Session {
public:
  /**
   *
   */
  explicit InferenceSession(const std::string &model_uri);

  virtual ~InferenceSession();

  common::Status Load(const void *model_data,
                      int model_data_len) BRT_MUST_USE_RESULT;

  common::Status Initialize() BRT_MUST_USE_RESULT;

  common::Status Run(const RunOptions &run_options,
                     const std::vector<std::string> &feed_names,
                     const std::vector<OrtValue> &feeds,
                     const std::vector<std::string> &output_names,
                     std::vector<OrtValue> *p_fetches,
                     const std::vector<OrtDevice> *p_fetches_device_info =
                         nullptr) BRT_MUST_USE_RESULT;

protected:
  bool IsInitialized() const;

  // names of model outputs used for quick validation.
  std::unordered_set<std::string> model_output_names_;

  // The file path of where the model was loaded.
  std::basic_string<BRTCHAR_T> model_location_;

  // The list of execution providers.
  execution_providers_;

private:
  BRT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(InferenceSession); // TODO expand it

  // Immutable state for each op in the model. Shared by all executors.
  // It has a dependency on execution_providers_.
  std::unique_ptr<SessionState> session_state_;

  ModelMetadata model_metadata_;
  std::unordered_set<std::string> required_inputs_;

  struct InputDefMetaData {
    InputDefMetaData(const NodeArg *node_arg0, MLDataType ml_data_type0,
                     TensorShape &&tensor_shape0)
        : node_arg(node_arg0), ml_data_type(ml_data_type0),
          tensor_shape(std::move(tensor_shape0)) {}
    const NodeArg *node_arg;
    MLDataType ml_data_type;
    TensorShape tensor_shape; // not applicable if the input is non-tensor type
  };

  std::unordered_map<std::string, InputDefMetaData> input_def_map_;
  OutputDefList output_def_list_;

  mutable brt::BrtMutex
      session_mutex_; // to ensure only one thread can invoke Load/Initialize
};

} // namespace brt
