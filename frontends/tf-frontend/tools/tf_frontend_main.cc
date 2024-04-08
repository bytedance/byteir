//===- tf_frontend_main.cc ------------------------------------------------===//
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

#include "absl/strings/str_split.h"

#include "mlir/IR/AsmState.h"           // from @llvm-project
#include "mlir/IR/MLIRContext.h"        // from @llvm-project
#include "mlir/Pass/PassManager.h"      // from @llvm-project
#include "mlir/Support/FileUtilities.h" // from @llvm-project
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/compiler/mlir/init_mlir.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate_cl.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/import_utils.h"

#include "tf_mlir_ext/pipelines/passes.h"
#include "utils/attributes.h"
#include "utils/graphdef_opt.h"
#include "utils/misc.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using namespace tensorflow;
using namespace mlir;

static llvm::cl::opt<std::string> input_filename(llvm::cl::Positional,
                                                 llvm::cl::desc("<input file>"),
                                                 llvm::cl::init("-"));

static llvm::cl::opt<std::string>
    output_filename("o", llvm::cl::desc("Output filename"),
                    llvm::cl::value_desc("filename"), llvm::cl::init("-"));

static llvm::cl::opt<int64_t> tfext_batch_size(
    "batch-size",
    llvm::cl::desc(
        "Specify batch size, remains not specified it is equal to -1."),
    llvm::cl::init(-1));

static llvm::cl::opt<bool> force_set_batch_size(
    "force-set-batch-size",
    llvm::cl::desc("override the first dimension with specified batch size "
                   "even if it is already set."),
    llvm::cl::init(false));

static llvm::cl::opt<bool> only_translate(
    "only-translate",
    llvm::cl::desc("Only translate tf graph to tf dialect, will not run the "
                   "subsequent conversion and transform passes."),
    llvm::cl::init(false));

static llvm::cl::opt<bool> only_grappler_optimize(
    "only-grappler-optimize",
    llvm::cl::desc("Only optimize the graph with grappler, output a pb file."),
    llvm::cl::init(false));

static llvm::cl::opt<bool> remove_control_flow(
    "remove-control-flow",
    llvm::cl::desc("Enable the experimental remove-control-flow pass"),
    llvm::cl::init(false));

static llvm::cl::opt<std::string>
    customcall_ops("customcall-ops",
                   llvm::cl::desc("convert ops to mhlo custom call."),
                   llvm::cl::init(""));

static llvm::cl::opt<std::string> input_name_and_shapes(
    "input-name-and-shapes",
    llvm::cl::desc("Specify some input's shapes. Ex. name0,2,3:name1,3,4"),
    llvm::cl::init(""));

static llvm::cl::opt<std::string>
    external_libs("external-libs",
                  llvm::cl::desc("additional lib's paths, seperated by comma."),
                  llvm::cl::init(""));

static llvm::cl::opt<std::string> reproducer_file(
    "reproduce-file",
    llvm::cl::desc("Generate a .mlir reproducer file at the given output path"
                   " if the pass manager crashes or fails"),
    llvm::cl::init(""));

static llvm::cl::opt<bool> staticalize_dynamic_shape(
    "staticalize-dynamic-shape",
    llvm::cl::desc("Aggresively and experimentally try to rewrite the dynamic "
                   "graph to a equivalent static graph"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> stop_after_convert_to_tf_dialect(
    "stop-after-convert-to-tf-dialect",
    llvm::cl::desc("pipeline stop after convert to tf dialect for debug"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> stop_after_rewrite_customcall(
    "stop-after-rewrite-customcall",
    llvm::cl::desc("pipeline stop after rewrite customcall ops for debug"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> keep_original_input_names(
    "keep-original-input-names",
    llvm::cl::desc("put original input names in main func as an ArrayAttr"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> set_assuming_to_be_true(
    "set-assuming-to-be-true",
    llvm::cl::desc("remove cstr_reshapable and cstr_broadcastable,"
                   "and remove assuming"),
    llvm::cl::init(true));

int main(int argc, char **argv) {
  tensorflow::InitMlir y(&argc, &argv);

  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "TF MLIR translation and opt driver\n");

  std::string error_message;
  auto output = mlir::openOutputFile(output_filename, &error_message);
  if (!output) {
    llvm::errs() << error_message << "\n";
    return 1;
  }

  mlir::MLIRContext context;
  std::unique_ptr<llvm::MemoryBuffer> input =
      mlir::openInputFile(input_filename, &error_message);
  if (!input) {
    llvm::errs() << error_message << "\n";
    return 1;
  }

  tensorflow::GraphDef input_graph_def;
  tensorflow::Status status = tensorflow::LoadProtoFromBuffer(
      {input->getBuffer().data(), input->getBuffer().size()}, &input_graph_def);
  if (!status.ok()) {
    llvm::errs() << "loading input graph to graph def failed."
                 << "\n";
    return 1;
  }

  std::vector<std::string> external_lib_path_vec;
  (void)tensorflow::ParseNodeNames(external_libs, external_lib_path_vec);
  for (const auto &path : external_lib_path_vec) {
    TF_Status *status_load = TF_NewStatus();
    TF_Library *lib_handle = TF_LoadLibrary(path.c_str(), status_load);
    if (!lib_handle) {
      llvm::errs() << "Load external library failed, path is: " << path << "\n";
      return 1;
    }
  }

  std::vector<std::string> input_names_vec;
  if (input_arrays != "") {
    (void)tensorflow::ParseNodeNames(input_arrays, input_names_vec);
  } else {
    input_names_vec = GetPlaceholderNames(input_graph_def);
  }

  // grappler optimization
  std::vector<std::string> output_array_vector;
  (void)tensorflow::ParseNodeNames(output_arrays, output_array_vector);
  tensorflow::GraphDef opted_graph_def = tensorflow::OptimizeGraphDef(
      input_graph_def, output_array_vector, input_names_vec);

  if (only_grappler_optimize) {
    output->os() << opted_graph_def.SerializeAsString();
    output->keep();
    return 0;
  }

  std::unordered_map<std::string, std::vector<int>> name2shape;
  (void)ParseStrToVectorIntMaps(input_name_and_shapes, name2shape);
  std::vector<tensorflow::InputInfo> new_input_infos =
      GetInputsByNameWithSpecifiedBatchSize(opted_graph_def, input_names_vec,
                                            tfext_batch_size,
                                            force_set_batch_size, name2shape);
  std::vector<std::string> new_input_dtypes_vec;
  std::vector<std::string> new_input_names_vec;
  std::vector<std::string> new_input_shapes_vec;
  for (const auto &info : new_input_infos) {
    new_input_names_vec.push_back(info.name);
    new_input_dtypes_vec.push_back(info.dtype);
    new_input_shapes_vec.push_back(info.shape_str);
  }
  input_arrays = tensorflow::join(new_input_names_vec, ",");
  input_dtypes = tensorflow::join(new_input_dtypes_vec, ",");
  input_shapes = tensorflow::join(new_input_shapes_vec, ":");

  tensorflow::GraphdefToMlirOptions options = {
      "",                 // debug_info_file
      "",                 // xla_compile_device_type
      prune_unused_nodes, // prune_unused_nodes
      false,              // convert_legacy_fed_inputs
      graph_as_function,  // graph_as_function
      false,              // upgrade_legacy
      false,              // enable_shape_inference
      false,              // unconditionally_use_set_output_shapes
      false               // enable_soft_placement
  };
  tsl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> module_or =
      tensorflow::GraphdefToMlirTranslateFunction(
          opted_graph_def.SerializeAsString(), input_arrays, input_dtypes,
          input_shapes, output_arrays, control_output_arrays, options,
          &context);
  if (!module_or.status().ok()) {
    return 1;
  }
  mlir::OwningOpRef<mlir::ModuleOp> module = std::move(module_or.value());

  if (only_translate) {
    module->print(output->os());
    output->keep();
    return 0;
  }

  std::vector<std::string> customcall_ops_array;
  (void)tensorflow::ParseNodeNames(customcall_ops, customcall_ops_array);
  mlir::PassManager tf_frontend_manager(module->getContext());
  if (!reproducer_file.empty()) {
    module->getContext()->disableMultithreading();
    tf_frontend_manager.enableCrashReproducerGeneration(reproducer_file, true);
  }

  std::unordered_map<std::string, Attribute> additional_main_func_attrs;

  if (keep_original_input_names) {
    llvm::SmallVector<Attribute> original_input_name_attrs;
    original_input_name_attrs.reserve(input_names_vec.size());
    for (const std::string &input_name : input_names_vec)
      original_input_name_attrs.push_back(
          StringAttr::get(module->getContext(), input_name));
    ArrayAttr original_input_name_array_attr =
        ArrayAttr::get(module->getContext(), original_input_name_attrs);
    additional_main_func_attrs[getTfOriginalInputNamesKey()] =
        original_input_name_array_attr;
  }

  tf_frontend_manager.addPass(
      ::mlir::tfext::createCustomizedTfToMhloPipelinePass(
          customcall_ops_array, remove_control_flow, staticalize_dynamic_shape,
          stop_after_convert_to_tf_dialect, stop_after_rewrite_customcall,
          additional_main_func_attrs, set_assuming_to_be_true));
  if (mlir::failed(tf_frontend_manager.run(*module))) {
    llvm::outs() << "tf frontend customized-tf-to-mhlo pipeline failed\n";
    return 1;
  }
  module->print(output->os());
  output->keep();
  return 0;
}
