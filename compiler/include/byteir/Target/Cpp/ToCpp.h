//===- ToCpp.h - Helpers to create C++ emitter ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines helpers to emit C++ code using the EmitC dialect.
//
//===----------------------------------------------------------------------===//
// Modifications Copyright 2022 ByteDance Ltd. and/or its affiliates.

#ifndef BYTEIR_TARGET_CPP_TOCPP_H
#define BYTEIR_TARGET_CPP_TOCPP_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"
#include <stack>

namespace byteir {

void registerToCppTranslation();

/// Translates the given operation to C++ code. The operation or operations in
/// the region of 'op' need almost all be in EmitC dialect. The parameter
/// 'declareVariablesAtTop' enforces that all variables for op results and block
/// arguments are declared at the beginning of the function.
mlir::LogicalResult translateToCpp(mlir::Operation *op, llvm::raw_ostream &os,
                                   bool declareVariablesAtTop = false);
} // namespace byteir

#endif // BYTEIR_TARGET_CPP_TOCPP_H
