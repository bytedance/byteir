/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------- OFCompilerUtils.cpp ------------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// Functions for adding passes and processing input files.
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/FileUtilities.h"
#include "onnx/onnx_pb.h"
#include "onnx/shape_inference/implementation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ToolOutputFile.h"

#include "third_party/onnx-mlir/src/Builder/FrontendDialectTransformer.hpp"
#include "third_party/onnx-mlir/src/Compiler/CompilerOptions.hpp"
#include "third_party/onnx-mlir/src/Compiler/CompilerUtils.hpp"

#include "stablehlo/api/PortableApi.h"
#include "stablehlo/dialect/Version.h"

#include "onnx-frontend/src/Compiler/OFCompilerOptions.hpp"
#include "onnx-frontend/src/Compiler/OFCompilerUtils.hpp"

#define DEBUG_TYPE "OFCompilerUtils"

#include <cassert>
#include <fstream>

namespace onnx_mlir {
void ImportFrontendModelInternal(onnx::ModelProto &model,
                                 mlir::MLIRContext &context,
                                 mlir::OwningOpRef<mlir::ModuleOp> &module,
                                 onnx_mlir::ImportOptions options);
} // namespace onnx_mlir

namespace onnx_frontend {

void ParseStrToVectorIntMaps(
    llvm::StringRef name_and_shapes,
    std::unordered_map<std::string, std::vector<int>> &name2shape) {
  // llvm split string into vector
  llvm::SmallVector<llvm::StringRef, 4> name_and_shape_vec;
  name_and_shapes.split(name_and_shape_vec, ':', -1, false);
  for (const auto &name_and_shape : name_and_shape_vec) {
    llvm::SmallVector<llvm::StringRef, 4> tokens;
    name_and_shape.split(tokens, ',', -1, false);
    if (tokens.size() <= 1) {
      printf("Not valid name and shape str: %s\n", name_and_shape.data());
      assert(false);
    }
    std::string name = tokens[0].str();
    std::vector<int> shape;
    for (unsigned int i = 1; i < tokens.size(); ++i) {
      shape.push_back(std::stoi(tokens[i].str()));
    }
    name2shape[name] = shape;
  }
}

void UpdateDimValue(onnx::ValueInfoProto &valueInfo,
                    const std::string &dimParam, int64_t value) {
  auto *type = valueInfo.mutable_type();
  LLVM_DEBUG(llvm::dbgs() << "value info name" << valueInfo.name() << "\n");
  if (type->value_case() != onnx::TypeProto::kTensorType)
    return;
  onnx::TensorShapeProto *shape = type->mutable_tensor_type()->mutable_shape();
  if (shape->dim_size() == 0)
    return;
  for (auto &dim : *(shape->mutable_dim())) {
    if (dim.has_dim_param() && dim.dim_param() == dimParam) {
      LLVM_DEBUG(llvm::dbgs() << "replace dynamic bs \n");
      dim.set_dim_value(value);
    }
  }
}

void ReplaceSymbolicDimValue(onnx::GraphProto *graph,
                             const std::string &dimParam, int64_t value) {
  for (onnx::ValueInfoProto &valueInfo : *(graph->mutable_value_info()))
    UpdateDimValue(valueInfo, dimParam, value);
  for (onnx::ValueInfoProto &valueInfo : *(graph->mutable_output()))
    UpdateDimValue(valueInfo, dimParam, value);
}

void SetBatchSize(onnx::ModelProto &model) {
  if (batchSize <= 0) {
    LLVM_DEBUG(llvm::dbgs() << "SetBatchSize() skipped for onnx pb\n");
    return;
  }

  onnx::GraphProto *graph = model.mutable_graph();
  std::string batchDimParam;
  bool hasDynamicBatchSize = false;
  bool sameStaticBatchSize = false;

  std::set<std::string> initializerNames;
  for (const auto &initializer : graph->initializer()) {
    const std::string &initializerName = initializer.name();
    initializerNames.insert(initializerName);
  }
  for (auto &input : *(graph->mutable_input())) {
    if (initializerNames.count(input.name()))
      continue;
    auto *type = input.mutable_type();
    if (type->value_case() != onnx::TypeProto::kTensorType) {
      continue;
    }
    auto *tensorType = type->mutable_tensor_type();
    if (!tensorType->has_shape()) {
      continue;
    }
    onnx::TensorShapeProto *shape = tensorType->mutable_shape();
    if (shape->dim_size() == 0) {
      continue;
    }
    auto *dim = shape->mutable_dim(0);
    bool isDynamic = (!dim->has_dim_value() || dim->dim_value() <= 0);
    if (isDynamic)
      hasDynamicBatchSize = true;
    else if (dim->dim_value() == batchSize)
      sameStaticBatchSize = true;
    if (isDynamic || forceSetBatchSize) {
      if (dim->has_dim_param()) {
        if (!batchDimParam.empty())
          assert(batchDimParam == dim->dim_param() &&
                 "mismatched batchsize dimparam among different inputs!");
        batchDimParam = dim->dim_param();
        LLVM_DEBUG(llvm::dbgs()
                   << "dynamic bs symbol: " << batchDimParam << "\n");
      }
      dim->set_dim_value(batchSize);
      LLVM_DEBUG(llvm::dbgs() << "bs of " << input.name() << " set to "
                              << batchSize << "\n");
    } else {
      LLVM_DEBUG(llvm::dbgs() << "bs of " << input.name() << " remains "
                              << dim->dim_value() << "\n");
    }
  }

  if (hasDynamicBatchSize &&
      !batchDimParam.empty()) { // replace dynamic batch size with dim_param
    LLVM_DEBUG(llvm::dbgs() << "replace dynamic bs\n");
    ReplaceSymbolicDimValue(graph, batchDimParam, batchSize);
  } else if (sameStaticBatchSize ||
             (!hasDynamicBatchSize &&
              !forceSetBatchSize)) { // same static batch size, or static bs w/o
                                     // forceSetBatchSize
    LLVM_DEBUG(llvm::dbgs() << "keep original bs\n");
  } else { // dynamic batch size without dim_param (?) or static bs with
           // forceSetBatchSize
    LLVM_DEBUG(llvm::dbgs() << "cannot keep value info, clear\n");
    graph->clear_value_info();
    for (auto &output : *(graph->mutable_output())) {
      auto *type = output.mutable_type();
      if (type->value_case() != onnx::TypeProto::kTensorType) {
        continue;
      }
      auto *tensorType = type->mutable_tensor_type();
      tensorType->clear_shape();
    }
  }
}

void SetInputShapes(onnx::ModelProto &model) {
  std::unordered_map<std::string, std::vector<int>> name2shape;
  ParseStrToVectorIntMaps(inputShapes, name2shape);
  onnx::GraphProto *graph = model.mutable_graph();
  for (auto &input : *(graph->mutable_input())) {
    if (!name2shape.count(input.name())) {
      continue;
    }
    auto *type = input.mutable_type();
    if (type->value_case() != onnx::TypeProto::kTensorType) {
      continue;
    }
    auto *tensorType = type->mutable_tensor_type();
    if (!tensorType->has_shape()) {
      continue;
    }
    onnx::TensorShapeProto *shape = tensorType->mutable_shape();
    if (shape->dim_size() == 0) {
      continue;
    }
    auto shapeVec = name2shape[input.name()];
    int rank = shapeVec.size();
    assert(shape->dim_size() == rank);
    for (int i = 0; i < rank; ++i) {
      auto *dim = shape->mutable_dim(i);
      dim->set_dim_value(shapeVec[i]);
      dim->clear_dim_param();
    }
  }
  for (auto &output : *(graph->mutable_output())) {
    auto *type = output.mutable_type();
    if (type->value_case() != onnx::TypeProto::kTensorType) {
      continue;
    }
    auto *tensorType = type->mutable_tensor_type();
    tensorType->clear_shape();
  }
}

namespace {
void traverse_onnx_graph_preorder(
    onnx::GraphProto &g, std::function<void(onnx::GraphProto &g)> callback) {
  callback(g);

  for (auto &node : *(g.mutable_node())) {
    for (auto &attr : *(node.mutable_attribute())) {
      if (attr.has_g()) {
        callback(*(attr.mutable_g()));
      }
    }
  }
}

void remove_invalid_dim_val_impl(onnx::GraphProto &g) {
  // remove invalid dim value in current `ValueInfoProto`.
  auto _remove_invalid_val = [](::onnx::ValueInfoProto &vi) {
    if (vi.has_type() && vi.type().has_tensor_type() &&
        vi.type().tensor_type().has_shape()) {
      onnx::TensorShapeProto *shape =
          vi.mutable_type()->mutable_tensor_type()->mutable_shape();
      for (int i = 0; i < shape->dim_size(); ++i) {
        auto dim = shape->mutable_dim(i);
        if (dim->has_dim_value() && dim->dim_value() < 1) {
          LLVM_DEBUG(llvm::dbgs() << "remove invalid dim val: " << vi.name()
                                  << "=" << dim->dim_value() << "\n");
          dim->Clear();
        }
      }
    }
  };

  for (auto &input : *(g.mutable_input()))
    _remove_invalid_val(input);

  for (auto &output : *(g.mutable_output()))
    _remove_invalid_val(output);

  for (auto &vi : *(g.mutable_value_info()))
    _remove_invalid_val(vi);
}
} // namespace

void RemoveInvalidDimValues(onnx::ModelProto &model) {
  onnx::GraphProto *graph = model.mutable_graph();
  traverse_onnx_graph_preorder(*graph, remove_invalid_dim_val_impl);
}

namespace {
std::string dirName(llvm::StringRef inputFilename) {
  llvm::SmallVector<char> path(inputFilename.begin(), inputFilename.end());
  llvm::sys::path::remove_filename(path);
  return std::string(path.data(), path.size());
}
} // namespace

// Return 0 on success, error number on failure.
int processInputFile(std::string inputFilename, mlir::MLIRContext &context,
                     mlir::OwningOpRef<mlir::ModuleOp> &module,
                     std::string *errorMessage) {
  // Decide if the input file is an ONNX model or a model specified
  // in MLIR. The extension of the file is the decider.
  std::string extension =
      inputFilename.substr(inputFilename.find_last_of(".") + 1);
  bool inputIsONNX = (extension == "onnx");

  if (!inputIsONNX) {
    *errorMessage = "Invalid input file '" + inputFilename +
                    "': An ONNX model (.onnx) needs to be provided.";
    return onnx_mlir::InvalidInputFile;
  }

  onnx_mlir::ImportOptions options;
  options.useOnnxModelTypes = onnx_mlir::useOnnxModelTypes;
  options.keepCustomOpTypes = onnx_mlir::keepCustomOpTypes;
  options.invokeOnnxVersionConverter = onnx_mlir::invokeOnnxVersionConverter;
  options.shapeInformation = onnx_mlir::shapeInformation;
  options.externalDataDir = dirName(inputFilename);

  onnx::ModelProto model;
  std::fstream input(inputFilename, std::ios::in | std::ios::binary);
  // check if the input file is opened
  if (!input.is_open()) {
    *errorMessage = "Unable to open or access " + inputFilename;
    return onnx_mlir::InvalidInputFileAccess;
  }

  auto parse_success = model.ParseFromIstream(&input);
  if (!parse_success) {
    *errorMessage = "Onnx Model Parsing Failed on " + inputFilename;
    return onnx_mlir::InvalidOnnxFormat;
  }
  RemoveInvalidDimValues(model);
  SetBatchSize(model);
  if (!inputShapes.empty()) {
    SetInputShapes(model);
  }
  onnx_mlir::ImportFrontendModelInternal(model, context, module, options);
  return onnx_mlir::CompilerSuccess;
}

void getStablehloSerialVersion(const std::string &inputVersion,
                               std::string &outputVersion) {
  auto inpVersion = mlir::vhlo::Version::fromString(inputVersion);
  if (mlir::failed(inpVersion)) {
    outputVersion = mlir::stablehlo::getCurrentVersion();
  } else {
    auto curVersion = mlir::vhlo::Version::getCurrentVersion();
    outputVersion = curVersion < *inpVersion ? curVersion.toString()
                                             : (*inpVersion).toString();
  }
}

// Return 0 on success, error code on error.
int outputModule(mlir::OwningOpRef<mlir::ModuleOp> &module,
                 llvm::raw_ostream &os, int64_t largeElementLimit = -1,
                 std::string serialVersion = "") {
  mlir::OpPrintingFlags flags;
  if (onnx_mlir::preserveLocations)
    flags.enableDebugInfo();
  if (largeElementLimit >= 0)
    flags.elideLargeElementsAttrs(largeElementLimit);
  if (serialVersion.empty())
    module->print(os, flags);
  else {
    // Generate stablehlo bytecode.
    std::string buffer;
    llvm::raw_string_ostream _os(buffer);
    module->print(_os, flags);
    if (mlir::failed(mlir::stablehlo::serializePortableArtifact(
            buffer, serialVersion, os))) {
      llvm::errs() << "Serialize to bytecode fail. version=" << serialVersion
                   << "\n";
      return onnx_mlir::CompilerFailure;
    }
  }
  return onnx_mlir::CompilerSuccess;
}

// Return 0 on success, error code on error.
int outputCode(mlir::OwningOpRef<mlir::ModuleOp> &module,
               std::string filenameWithExt, int64_t largeElementLimit,
               std::string serialVersion = "") {
  std::string errorMessage;
  auto output = mlir::openOutputFile(filenameWithExt, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return onnx_mlir::InvalidOutputFileAccess;
  }
  auto retCode =
      outputModule(module, output->os(), largeElementLimit, serialVersion);
  output->keep();
  return retCode;
  // return onnx_mlir::CompilerSuccess;
}

// Return 0 on success, error code on failure.
int emitOutput(mlir::OwningOpRef<mlir::ModuleOp> &module,
               std::string outputFilename,
               onnx_frontend::EmissionTargetType emissionTarget, bool emitElide,
               std::string serialVersion) {
  if (emissionTarget == onnx_frontend::EmitONNXIR ||
      emissionTarget == onnx_frontend::EmitStablehloIR) {
    if (emitElide) {
      return outputCode(module, outputFilename,
                        /*largeElementLimit=*/100, serialVersion);
    }
    return outputCode(module, outputFilename, /*largeElementLimit=*/-1,
                      serialVersion);
  }
  return onnx_mlir::InvalidCompilerOption;
}

// Return 0 on success, error code on error.
int compileModule(mlir::OwningOpRef<mlir::ModuleOp> &module,
                  mlir::PassManager &pm, std::string outputFilename,
                  onnx_frontend::EmissionTargetType emissionTarget,
                  bool emitElide, std::string serialVersion) {
  bool runFailure = mlir::failed(pm.run(*module));
  int outputStatus = emitOutput(module, outputFilename, emissionTarget,
                                emitElide, serialVersion);
  if (runFailure)
    return onnx_mlir::CompilerFailure;
  return outputStatus;
}

} // namespace onnx_frontend
