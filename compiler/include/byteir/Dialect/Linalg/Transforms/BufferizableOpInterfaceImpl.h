//===- BufferizableOpInterfaceImpl.h - Impl. of BufferizableOpInterface ---===//
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

#ifndef BYTEIR_DIALECT_LINALG_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H
#define BYTEIR_DIALECT_LINALG_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"

namespace mlir {
class DialectRegistry;

namespace linalg_ext {

struct LinalgExtBufferizableOpInterfaceImpl {
  bool
  bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                         const bufferization::AnalysisState & /* state*/) const;

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const bufferization::AnalysisState &state) const;

  bufferization::AliasingOpOperandList
  getAliasingOpOperands(Operation *op, OpResult opResult,
                        const bufferization::AnalysisState &) const;

  bufferization::AliasingOpResultList
  getAliasingOpResults(Operation *op, OpOperand &opOperand,
                       const bufferization::AnalysisState &) const;

  bufferization::BufferRelation
  bufferRelation(Operation *op, OpResult opResult,
                 const bufferization::AnalysisState &state) const;

  LogicalResult
  bufferize(Operation *op, RewriterBase &rewriter,
            const bufferization::BufferizationOptions &options) const;
};

template <typename OpTy>
struct LinalgExtBufferizableOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<
          LinalgExtBufferizableOpInterface<OpTy>, OpTy>,
      public LinalgExtBufferizableOpInterfaceImpl {
  using LinalgExtBufferizableOpInterfaceImpl::bufferize;
  using LinalgExtBufferizableOpInterfaceImpl::bufferizesToMemoryRead;
  using LinalgExtBufferizableOpInterfaceImpl::bufferizesToMemoryWrite;
  using LinalgExtBufferizableOpInterfaceImpl::bufferRelation;
  using LinalgExtBufferizableOpInterfaceImpl::getAliasingOpOperands;
  using LinalgExtBufferizableOpInterfaceImpl::getAliasingOpResults;
};

void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);
} // namespace linalg_ext
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_BUFFERIZABLEOPINTERFACEIMPL_H
