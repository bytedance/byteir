//===- Einsum.cpp ---------------------------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/Shape/IR/ShapeExtOps.h"
#include "byteir/Dialect/mhlo/DynamicShapeOpRegister/Register.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/Sequence.h"

#define DEBUG_TYPE "dynamic-shape-op-register"

using namespace mlir;

#define K_INITIAL -999

namespace {

// valid label: [a-zA-Z]
static constexpr uint8_t kTotalLabels = 52;
// Code used to identify ELLIPSIS ("...")
static constexpr uint8_t kEllipsis = 52;

bool einsumCheckLabel(unsigned char label) { return std::isalpha(label); }

uint8_t einsumLabelToIndex(unsigned char label) {
  constexpr uint8_t kNumOfLetters = 'z' - 'a' + 1;
  return std::isupper(label) ? label - 'A' : kNumOfLetters + (label - 'a');
}

unsigned char einsumIndexToLabel(uint8_t index) {
  constexpr uint8_t kNumOfLetters = 'z' - 'a' + 1;
  return index < kNumOfLetters ? index + 'A' : index - kNumOfLetters + 'a';
}

struct EinsumParseContext {
  //===----------------------------------------------------------------------===//
  // basic info
  //===----------------------------------------------------------------------===//
  // einsum expression's form: $lhs[->$rhs]
  llvm::StringRef lhs;
  llvm::StringRef rhs;
  bool isImplicit = false;

  //===----------------------------------------------------------------------===//
  // operands info
  //===----------------------------------------------------------------------===//
  // num of operands of this equation
  size_t numOps = 0;
  // the maximum dimention num that the ellipsis covers
  int64_t ellNumDim = 0;
  // list of labels of each operand
  llvm::SmallVector<llvm::SmallVector<uint8_t>> opLabels;
  // the number of appearances of each label in the operands
  llvm::SmallVector<int64_t> labelCount;

  //===----------------------------------------------------------------------===//
  // ouptuts info
  //===----------------------------------------------------------------------===//
  // output dim size
  int64_t outSize = 0;
  // total dim size, including outSize + size of contraction labels
  int64_t totalSize = 0;
  // Start index of ellipsis dimensions in the permuted shape
  int64_t ellIndex = 0;
  // We want to align the dimensions of every input tensor to have
  // shape out_dims + sum_dims. For this, we create a mapping of label
  // to index into the permuted shape.
  llvm::SmallVector<int64_t> labelPermIndex;

  //===----------------------------------------------------------------------===//
  // operands permutation info:
  // before computation, operands dimensions are unsqueezed and permuted
  // to have the same number of dimensions. We can't do these operations in the
  // IR, but we can record operations need to be done
  //===----------------------------------------------------------------------===//
  // for each operand, map the normalized permuted dimensions to original
  // dimensions
  llvm::SmallVector<llvm::SmallVector<int64_t>> permuteDim;
};

LogicalResult einsumParseBasicInfo(Operation *op, EinsumParseContext *ctx) {
  llvm::StringRef einsumConfig;

  // parse einsum config
  if (auto customCallOp = llvm::dyn_cast<mhlo::CustomCallOp>(op)) {
    // auto attrs = customCallOp->getAttrDictionary();
    llvm::errs() << "mhlo::CustomCallOp for einsum not supported yet";
    return failure();
  } else if (auto einsumOp = llvm::dyn_cast<mhlo::EinsumOp>(op)) {
    einsumConfig = einsumOp.getEinsumConfig();
  } else {
    llvm::errs() << "op not mhlo::CustomCallOp or mhlo::EinsumOp";
    return failure();
  }

  const auto arrowPos = einsumConfig.find("->");
  bool isImplicit = arrowPos == llvm::StringRef::npos;
  ctx->lhs = einsumConfig.substr(0, arrowPos);
  ctx->isImplicit = isImplicit;
  if (!isImplicit) {
    ctx->rhs = einsumConfig.substr(arrowPos + 2);
  } else {
    ctx->rhs = "";
  }
  return success();
}

LogicalResult einsumParseOperandsInfo(Operation *op, EinsumParseContext *ctx) {
  const auto numOps = op->getNumOperands();
  llvm::SmallVector<llvm::SmallVector<uint8_t>> opLabels(numOps);

  bool foundEll = false;
  size_t currOp = 0;
  for (auto i = decltype(ctx->lhs.size()){0}; i < ctx->lhs.size(); ++i) {
    const unsigned char label = ctx->lhs[i];
    switch (label) {
    case ' ': {
      // Ignore spaces
      break;
    }
    case '.': {
      if (foundEll) {
        llvm::errs() << "einsum: found \'.'\' for operand " << currOp
                     << " for which an ellipsis was already found\n";
        return failure();
      }
      if (!(i + 2 < ctx->lhs.size() && ctx->lhs[++i] == '.' &&
            ctx->lhs[++i] == '.')) {
        llvm::errs() << "einsum: found \'.\' for operand " << currOp
                     << " that is not part of any ellipsis\n";
        return failure();
      }
      opLabels[currOp].push_back(kEllipsis);
      foundEll = true;
      break;
    }
    case ',': {
      // Move onto next operand
      ++currOp;
      if (currOp >= numOps) {
        llvm::errs() << "einsum: fewer operands were provided than specified "
                        "in the equation\n";
        return failure();
      }
      foundEll = false;
      break;
    }
    default: {
      // Parse label
      if (!einsumCheckLabel(label)) {
        llvm::errs()
            << "einsum: invalid subscript given at index " << i
            << " in the equation string, subscripts must be in [a-zA-z]\n";
        return failure();
      }
      opLabels[currOp].push_back(einsumLabelToIndex(label));
    }
    }
  }

  if (currOp != numOps - 1) {
    llvm::errs() << "einsum: more operands were provided than specified in the "
                    "equation\n";
    return failure();
  }

  llvm::SmallVector<int64_t> labelCount(kTotalLabels, 0);

  // The maximum number of dimensions covered by any ellipsis, needed when
  // unsqueezing missing dimensions from operands to permute and broadcast
  int64_t ellNumDim = 0;

  // Compute label frequency and number of dimensions covered by ellipsis
  // We do this after parsing labels to make it more readable and simpler
  // to compute the number of dimensions covered by ellipsis.
  for (const auto i : llvm::iota_range<int>(0, numOps, /*Inclusive=*/false)) {
    const auto operand = op->getOperand(i);
    auto operandType = operand.getType().cast<RankedTensorType>();
    const auto &labels = opLabels[i];
    const auto ndims = operandType.getRank();
    int64_t nlabels = static_cast<int64_t>(labels.size());
    bool hasEllipsis = false;

    for (const auto label : labels) {
      if (label == kEllipsis) {
        --nlabels;
        hasEllipsis = true;
        ellNumDim = std::max(ellNumDim, ndims - nlabels);
      } else {
        ++labelCount[label];
      }
    }
    if (!(hasEllipsis ? nlabels <= ndims : nlabels == ndims)) {
      llvm::errs() << "einsum: the number of subscripts in the equation ("
                   << nlabels
                   << (hasEllipsis
                           ? ") is more than the number of dimensions ("
                           : ") does not match the number of dimensions (")
                   << ndims << ") for operand " << i
                   << (hasEllipsis ? "\n" : " and no ellipsis was given\n");
      return failure();
    }
  }

  // update context
  ctx->numOps = numOps;
  ctx->ellNumDim = ellNumDim;
  ctx->opLabels.swap(opLabels);
  ctx->labelCount.swap(labelCount);
  return success();
}

LogicalResult einsumParseOuptutInfo(Operation *op, EinsumParseContext *ctx) {
  // Current index in the permuted shape
  int64_t permIndex = 0;
  // Start index of ellipsis dimensions in the permuted shape
  int64_t ellIndex = 0;
  bool foundEll = false;
  llvm::SmallVector<int64_t> labelPermIndex(kTotalLabels, K_INITIAL);

  if (ctx->isImplicit) {
    // Implicit output is ellipsis (...) + labels seen only once
    permIndex = ctx->ellNumDim;
    foundEll = true;
    for (const auto label :
         llvm::iota_range<uint8_t>(0, kTotalLabels, /*Inclusive=*/false)) {
      if (ctx->labelCount[label] == 1) {
        labelPermIndex[label] = permIndex++;
      }
    }
  } else {
    for (auto i = decltype(ctx->rhs.size()){0}; i < ctx->rhs.size(); ++i) {
      const unsigned char label = ctx->rhs[i];
      switch (label) {
      case ' ': {
        // Ignore spaces
        break;
      }
      case '.': {
        if (foundEll) {
          llvm::errs()
              << "einsum: found \'.\' for output but an ellipsis (...) "
                 "was already found\n";
          return failure();
        }
        if (!(i + 2 < ctx->rhs.size() && ctx->rhs[++i] == '.' &&
              ctx->rhs[++i] == '.')) {
          llvm::errs() << "einsum: found \'.\' for output that is not part of "
                          "any ellipsis (...)\n";
          return failure();
        }
        ellIndex = permIndex;
        permIndex += ctx->ellNumDim;
        foundEll = true;
        break;
      }
      default: {
        if (!einsumCheckLabel(label)) {
          llvm::errs()
              << "einsum: invalid subscript given at index "
              << ctx->lhs.size() + 2 + i
              << " in the equation string, subscripts must be in [a-zA-Z]\n";
          return failure();
        }
        const auto index = einsumLabelToIndex(label);

        if (!(ctx->labelCount[index] > 0 &&
              labelPermIndex[index] == K_INITIAL)) {
          llvm::errs() << "einsum: output subscript " << label
                       << (labelPermIndex[index] > K_INITIAL
                               ? " appears more than once in the output\n"
                               : " does not appear in the equation for any "
                                 "input operand\n");
          return failure();
        }
        labelPermIndex[index] = permIndex++;
      }
      }
    }
  }

  // Save output size before adding contraction dims (dims to sum out)
  const int64_t outSize = permIndex;

  // If ellipsis is not part of the output, add to contraction dimensions
  if (!foundEll) {
    ellIndex = permIndex;
    permIndex += ctx->ellNumDim;
  }

  // Add contraction labels (labels not present in output)
  for (const auto label : llvm::iota_range<uint8_t>(0, kTotalLabels,
                                                    /*Inclusive=*/false)) {
    if (ctx->labelCount[label] > 0 && labelPermIndex[label] == K_INITIAL) {
      labelPermIndex[label] = permIndex++;
    }
  }
  // update context
  ctx->outSize = outSize;
  ctx->totalSize = permIndex;
  ctx->ellIndex = ellIndex;
  ctx->labelPermIndex.swap(labelPermIndex);
  return success();
}

LogicalResult einsumParseOperandsPermuteInfo(Operation *op,
                                             EinsumParseContext *ctx) {

  llvm::SmallVector<llvm::SmallVector<int64_t>> permuteDim(ctx->numOps);
  for (const auto i :
       llvm::iota_range<int>(0, ctx->numOps, /*Inclusive=*/false)) {
    llvm::SmallVector<int64_t> permShape(ctx->totalSize, K_INITIAL);
    llvm::SmallVector<int64_t> labelDim(kTotalLabels, K_INITIAL);

    // const auto operand = op->getOperand(i);
    const auto &labels = ctx->opLabels[i];

    // current dimension index of the operand
    int64_t j = 0;
    for (const auto &label : labels) {
      if (label == kEllipsis) {
        // Add missing dimensions covered by the ellipsis
        // const auto numMissingDim =
        //     ell_num_dim - (original_sizes.size() - labels.size() + 1);
        // for (const auto k : c10::irange(num_missing_dim)) {
        //   (void)k; // Suppress unused warning
        //   operand = operand.unsqueeze(j);
        // }
        for (const auto k :
             llvm::iota_range<int>(0, ctx->ellNumDim, /*Inclusive=*/false)) {
          permShape[ctx->ellIndex + k] = j++;
        }
      } else if (labelDim[label] != K_INITIAL) {
        llvm::errs() << "einsum: subscript " << einsumIndexToLabel(label)
                     << " is repeated for operand, which is not supported to "
                        "insert shape constraints\n";
        return failure();
      } else {
        // Lookup output index for label
        labelDim[label] = j;
        const auto index = ctx->labelPermIndex[label];
        permShape[index] = j++;
      }
    }
    permuteDim[i].swap(permShape);
  }
  // update context
  ctx->permuteDim.swap(permuteDim);
  return success();
}

} // namespace

void mlir::registerEinsumShapeConstraints() {
  // reference to einsum op logic:
  // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Linear.cpp
  auto insertShapeConstraintFunc = [](Operation *op, OpBuilder &builder) {
    //===----------------------------------------------------------------------===//
    // parse the einsum op to get necessary information in ctx
    //===----------------------------------------------------------------------===//
    EinsumParseContext ctx;
    if (failed(einsumParseBasicInfo(op, &ctx))) {
      return failure();
    }

    if (failed(einsumParseOperandsInfo(op, &ctx))) {
      return failure();
    }

    if (failed(einsumParseOuptutInfo(op, &ctx))) {
      return failure();
    }

    if (failed(einsumParseOperandsPermuteInfo(op, &ctx))) {
      return failure();
    }

    //===----------------------------------------------------------------------===//
    // insert shape constraints
    //===----------------------------------------------------------------------===//

    // init builder position
    builder.setInsertionPointAfter(op);

    auto getDimSize = [](const Value &operand, int dimIdx) {
      auto type = operand.getType().cast<RankedTensorType>();
      return type.getDimSize(dimIdx);
    };

    // iterate dimensions
    for (const auto dim :
         llvm::iota_range<int>(0, ctx.totalSize, /*Inclusive=*/false)) {
      // the index of operand whose #dim-th dimension size is used for
      // shape_ext.meet with all other operands and result
      int baseOperandIndex = K_INITIAL;

      for (const auto i :
           llvm::iota_range<int>(0, ctx.numOps, /*Inclusive=*/false)) {
        Value operand = op->getOperand(i);
        const auto originDim = ctx.permuteDim[i][dim];
        if (originDim < 0) {
          // #dim not from this operand, skip
          continue;
        }
        const auto dimSize = getDimSize(operand, originDim);
        if (baseOperandIndex == K_INITIAL || dimSize > 0) {
          baseOperandIndex = i;
        }
      }
      if (baseOperandIndex < 0) {
        llvm::errs() << "einsum: no base operand to dim " << dim << " found\n";
        return failure();
      }
      Value baseOperand = op->getOperand(baseOperandIndex);
      const auto baseOperandOriginDim = ctx.permuteDim[baseOperandIndex][dim];
      Value baseDim = builder.create<tensor::DimOp>(op->getLoc(), baseOperand,
                                                    baseOperandOriginDim);

      for (const auto i :
           llvm::iota_range<int>(0, ctx.numOps, /*Inclusive=*/false)) {
        if (i == baseOperandIndex) {
          continue;
        }
        Value operand = op->getOperand(i);
        const auto originDim = ctx.permuteDim[i][dim];
        if (originDim < 0) {
          continue;
        }
        Value dim =
            builder.create<tensor::DimOp>(op->getLoc(), operand, originDim);
        builder.create<shape_ext::MeetOp>(op->getLoc(), baseDim, dim);
      }

      if (dim < ctx.outSize) {
        Value result = op->getResult(0);
        Value resultDim =
            builder.create<tensor::DimOp>(op->getLoc(), result, dim);
        builder.create<shape_ext::MeetOp>(op->getLoc(), baseDim, resultDim);
      }
    }

    return success();
  };

  static InsertShapeConstraintRegistration shapeRegister(
      mhlo::EinsumOp::getOperationName(), insertShapeConstraintFunc);
}
