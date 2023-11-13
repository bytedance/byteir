//===- byteir-translate.cpp - ByteIR's MLIR Codegen Driver-----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for byteir-translate for when built as standalone binary.
//
//===----------------------------------------------------------------------===//
// Modifications Copyright 2022 ByteDance Ltd. and/or its affiliates.

#include "byteir/Target/CUDA/ToCUDA.h"
#include "byteir/Target/Cpp/ToCpp.h"
#include "byteir/Target/LLVM/ToLLVMBC.h"
#include "byteir/Target/PTX/ToPTX.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

using namespace mlir;

int main(int argc, char **argv) {
  registerAllTranslations();
  registerToPTXTranslation();
  byteir::registerToCppTranslation();
  byteir::registerToCUDATranslation();
  byteir::registerToLLVMBCTranslation();

  return failed(
      mlirTranslateMain(argc, argv, "ByteIR Translation Testing Tool"));
}
