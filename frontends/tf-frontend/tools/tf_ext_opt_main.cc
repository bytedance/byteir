//===- tf_ext_opt_main.cc -------------------------------------*--- C++ -*-===//
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
// Some code comes from Tensorflow project
// Original license:
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lhlo/transforms/passes.h"
#include "mhlo/IR/register.h"
#include "mhlo/transforms/passes.h"
#include "mlir/Dialect/Shape/IR/Shape.h"     // from @llvm-project
#include "mlir/InitAllDialects.h"            // from @llvm-project
#include "mlir/InitAllPasses.h"              // from @llvm-project
#include "mlir/Tools/mlir-opt/MlirOptMain.h" // from @llvm-project
#include "tensorflow//compiler/mlir/tensorflow/transforms/tf_saved_model_passes.h"
#include "tensorflow/compiler/mlir/init_mlir.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/test_passes.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/passes.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"
#include "tensorflow/core/platform/init_main.h"

#include "byteir/Dialect/Ace/AceDialect.h"
#include "tf_mlir_ext/pipelines/passes.h"
#include "tf_mlir_ext/transforms/passes.h"
#include "tf_mlir_ext/transforms/process_dynamic_stitch_as_static.h"

int main(int argc, char **argv) {
  tensorflow::InitMlir y(&argc, &argv);

  mlir::registerAllPasses();
  mlir::registerTensorFlowPasses();
  mlir::TFDevice::registerTensorFlowDevicePasses();
  mlir::tf_saved_model::registerTensorFlowSavedModelPasses();
  mlir::mhlo::registerAllMhloPasses();
  mlir::lmhlo::registerAllLmhloPasses();
  // These are in compiler/mlir/xla and not part of the above MHLO passes.
  mlir::mhlo::registerTfXlaPasses();
  mlir::mhlo::registerLegalizeTFPass();
  // mlir::mhlo::registerLegalizeTfTypesPassPass();
  mlir::TFL::registerTensorFlowLitePasses();
  mlir::tf_test::registerTensorFlowTestPasses();

  mlir::registerTensorFlowExtensionPasses();
  mlir::registerTensorFlowExtensionPipelinesPasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::RegisterAllTensorFlowDialects(registry);
  mlir::mhlo::registerAllMhloDialects(registry);
  registry.insert<mlir::ace::AceDialect>(); // register ace dialect
  registry.insert<mlir::shape::ShapeDialect>();
  registry.insert<mlir::kernel_gen::tf_framework::TFFrameworkDialect>();
  return mlir::failed(
      mlir::MlirOptMain(argc, argv, "TensorFlow pass driver\n", registry));
}
