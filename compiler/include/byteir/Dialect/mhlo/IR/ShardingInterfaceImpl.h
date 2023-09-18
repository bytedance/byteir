//===- ShardingInterfaceImpl.h - ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_MHLO_IR_SHARDINGINTERFACEIMPL_H
#define BYTEIR_DIALECT_MHLO_IR_SHARDINGINTERFACEIMPL_H

#include "mlir/IR/Dialect.h"

namespace mlir {

class DialectRegistry;

namespace mhlo {

void registerShardingInterfaceExternalModels(DialectRegistry &registry);

} // namespace mhlo
} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_IR_SHARDINGINTERFACEIMPL_H
