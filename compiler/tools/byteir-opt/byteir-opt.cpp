//===- byteir-opt.cpp - ByteIR's MLIR Optimizer Driver---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for byteir-opt for when built as standalone binary.
//
//===----------------------------------------------------------------------===//
// Modifications Copyright 2022 ByteDance Ltd. and/or its affiliates.

#include "byteir/Conversion/Passes.h"
#include "byteir/Dialect/Ace/AceDialect.h"
#include "byteir/Dialect/Ace/Passes.h"
#include "byteir/Dialect/Affine/Passes.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "byteir/Dialect/Byre/Passes.h"
#include "byteir/Dialect/Ccl/IR/CclOps.h"
#include "byteir/Dialect/Lace/LaceDialect.h"
#include "byteir/Dialect/Linalg/IR/LinalgExtOps.h"
#include "byteir/Dialect/Linalg/Passes.h"
#include "byteir/Dialect/Linalg/TransformOps/LinalgExtTransformOps.h"
#include "byteir/Dialect/MemRef/Passes.h"
#include "byteir/Dialect/SCF/Passes.h"
#include "byteir/Dialect/Shape/IR/ShapeExtOps.h"
#include "byteir/Dialect/Shape/Passes.h"
#include "byteir/Dialect/Tensor/IR/TilingInterfaceImpl.h"
#include "byteir/Dialect/Transform/IR/TransformExtOps.h"
#include "byteir/Dialect/Transform/Passes.h"
#include "byteir/Dialect/mhlo/Passes.h"
#include "byteir/Pipelines/InitAllPipelines.h"
#include "byteir/Transforms/Passes.h"
#include "gml_st/IR/gml_st_ops.h"
#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/test_passes.h"
#include "lhlo/IR/lhlo_ops.h"
#include "lhlo/transforms/passes.h"
#include "lhlo_gpu/IR/lhlo_gpu_ops.h"
#include "mhlo/IR/hlo_ops.h"
#include "mhlo/IR/register.h"
#include "mhlo/transforms/passes.h"
#include "mlir-hlo/Transforms/gpu_passes.h"
#include "mlir-hlo/Transforms/passes.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "stablehlo/dialect/Register.h"
#include "thlo/IR/thlo_ops.h"
#include "thlo/transforms/passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;
using namespace mlir;

namespace byteir {
namespace test {
void registerTestMhloCanonicalizeExtPass();
void registerTestConvertFuncToCustomCallPass();
void registerTestConvertInsertionPass();
void registerTestCustomConvertPass();
void registerTestDTypeConversionPass();
void registerTestFuncArgRearrangementPass();
void registerTestPrintArgSideEffectPass();
void registerTestPrintLivenessPass();
void registerTestPrintUseRangePass();
void registerTestPrintSymbolicShapePass();
void registerTestPrintShapeAnalysisPass();
void registerTestByreOpInterfacePass();
void registerTestBroadcastDenseElementsAttrPass();
} // namespace test
} // namespace byteir

#ifdef BYTEIR_INCLUDE_TESTS
void registerTestPasses() {
  byteir::test::registerTestMhloCanonicalizeExtPass();
  byteir::test::registerTestConvertFuncToCustomCallPass();
  byteir::test::registerTestConvertInsertionPass();
  byteir::test::registerTestCustomConvertPass();
  byteir::test::registerTestDTypeConversionPass();
  byteir::test::registerTestFuncArgRearrangementPass();
  byteir::test::registerTestPrintArgSideEffectPass();
  byteir::test::registerTestPrintLivenessPass();
  byteir::test::registerTestPrintUseRangePass();
  byteir::test::registerTestPrintSymbolicShapePass();
  byteir::test::registerTestPrintShapeAnalysisPass();
  byteir::test::registerTestByreOpInterfacePass();
  byteir::test::registerTestBroadcastDenseElementsAttrPass();
}
#endif

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::hlo::registerLMHLOTransformsPasses();
  mlir::registerLMHLOGPUTransformsPasses();
  mlir::mhlo::registerAllMhloPasses();
  mlir::lmhlo::registerAllLmhloPasses();
  mlir::thlo::registerAllThloPasses();
  mlir::gml_st::registerGmlStPasses();
  mlir::gml_st::registerGmlStTestPasses();

  registerByteIRConversionPasses();
  registerByteIRTransformsPasses();
  registerByteIRAcePasses();
  registerByteIRAffinePasses();
  registerByteIRByrePasses();
  registerByteIRLinalgPasses();
  registerByteIRMemRefPasses();
  registerByteIRMhloPassesExt();
  registerByteIRSCFPasses();
  registerByteIRShapePasses();
  registerByteIRTransformPasses();

  registerAllByteIRCommonPipelines();
  registerAllByteIRGPUPipelines();
  registerAllByteIRHostPipelines();

#ifdef BYTEIR_INCLUDE_TESTS
  registerTestPasses();
#endif

  DialectRegistry registry;
  registerAllDialects(registry);

  // register ByteIR's dialects here
  registry.insert<mlir::ace::AceDialect>();
  registry.insert<mlir::byre::ByreDialect>();
  registry.insert<mlir::ccl::CclDialect>();
  registry.insert<mlir::mhlo::MhloDialect>();
  registry.insert<mlir::lace::LaceDialect>();
  registry.insert<mlir::lmhlo::LmhloDialect>();
  registry.insert<mlir::shape_ext::ShapeExtDialect>();
  registry.insert<mlir::linalg_ext::LinalgExtDialect>();

  // register extension
  linalg_ext::registerTransformDialectExtension(registry);
  transform_ext::registerTransformDialectExtension(registry);
  tensor_ext::registerTilingInterfaceExternalModels(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "ByteIR pass driver\n", registry,
                        /*preloadDialectsInContext=*/false));
}
