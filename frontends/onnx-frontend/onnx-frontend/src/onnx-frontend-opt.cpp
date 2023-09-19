//===- onnx-frontend-opt.cpp ----------------------------------------------===//
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

#include <llvm/Support/InitLLVM.h>
#include <mlir/IR/Dialect.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>
#include <mlir/Transforms/Passes.h>

#include "third_party/onnx-mlir/src/Compiler/CompilerUtils.hpp"
#include "third_party/onnx-mlir/src/Dialect/ONNX/ONNXDialect.hpp"
#include "third_party/onnx-mlir/src/Tools/onnx-mlir-opt/RegisterPasses.hpp"

#include "onnx-frontend/src/Conversion/OFPasses.hpp"

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  mlir::registerTransformsPasses();
  onnx_mlir::registerPasses(onnx_mlir::O0);
  onnx_frontend::registerOFConversionPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::shape::ShapeDialect>();
  registry.insert<mlir::ONNXDialect>();
  registry.insert<mlir::mhlo::MhloDialect>();

  return mlir::failed(mlir::MlirOptMain(
      argc, argv, "ONNX-Frontend modular optimizer driver\n", registry));
}
