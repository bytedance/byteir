//===- Util.cpp -----------------------------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/mhlo/Util/Util.h"
#include "byteir/Dialect/mhlo/DynamicShapeOpRegister/Register.h"
#include "byteir/Dialect/mhlo/Util/ShapeInferUtil.h"
#include "byteir/Utils/Utils.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::mhlo;

#define K_INITIAL -999

bool mlir::isMhlo(Operation *op) {
  Dialect *dialect = op->getDialect();
  return dialect && isa<MhloDialect>(dialect);
}

bool mlir::isSplatMhloConstant(Operation *op) {
  if (auto constOp = dyn_cast_or_null<mhlo::ConstantOp>(op)) {
    return constOp.getValue().isSplat();
  }
  return false;
}

bool mlir::isSplatMhloConstantLike(Operation *op) {
  return isSplatMhloConstant(op) || isa_and_nonnull<mhlo::IotaOp>(op);
}

bool mlir::isMhloConstantLike(Operation *op) {
  if (!op)
    return false;
  return isa<mhlo::ConstantOp>(op) || isa<mhlo::IotaOp>(op);
}

bool mlir::isDeepMhloFoldable(Operation *op) {
  auto check = [](Operation *op) {
    if (!op)
      return false;
    return isMhlo(op);
  };
  return deepCheck(op, check);
}

bool mlir::isSplatMhloConstantValue(Value val) {
  return isSplatMhloConstant(val.getDefiningOp());
}

bool mlir::isSplatMhloConstantValue(Operation *op, int64_t splat_val) {
  if (auto constOp = dyn_cast_or_null<mhlo::ConstantOp>(op)) {
    // only handle DenseIntElementsAttr for now
    // TODO: extend it
    if (auto denseIntE = dyn_cast<DenseIntElementsAttr>(constOp.getValue())) {
      return isSplatValue(denseIntE, splat_val);
    }
  }
  return false;
}

bool mlir::isSplatMhloConstantValue(Operation *op, double splat_val) {
  if (auto constOp = dyn_cast_or_null<mhlo::ConstantOp>(op)) {
    // only handle DenseFPElementsAttr for now
    // TODO: extend it
    if (auto denseFPE = dyn_cast<DenseFPElementsAttr>(constOp.getValue())) {
      return isSplatValue(denseFPE, splat_val);
    }
  }
  return false;
}

bool mlir::isDenseMhloConstantValue(Value val) {
  if (auto constOp = val.getDefiningOp<mhlo::ConstantOp>()) {
    return llvm::isa<DenseElementsAttr>(constOp.getValue());
  }
  return false;
}

bool mlir::isSplatMhloConstantValue(Value val, int64_t splat_val) {
  return isSplatMhloConstantValue(val.getDefiningOp(), splat_val);
}

bool mlir::isSplatMhloConstantValue(Value val, double splat_val) {
  return isSplatMhloConstantValue(val.getDefiningOp(), splat_val);
}

// Return true if op is a regular reduce/reduce_window op, like reduce
// max/min/sum/any
template <typename RegionOp, typename Op> bool mlir::isRegularReduceOp(Op op) {
  if (op.getInputs().size() != 1 || op.getInitValues().size() != 1 ||
      op.getResults().size() != 1) {
    return false;
  }
  if (!isBlockSingleOp<RegionOp>(&op.getBody().front())) {
    return false;
  }

  SplatElementsAttr initValue;
  if (!matchPattern(op.getInitValues()[0], m_Constant(&initValue))) {
    return false;
  }

  if (std::is_same_v<RegionOp, mhlo::AddOp> && isZeroAttribute(initValue)) {
    return true;
  } else if (std::is_same_v<RegionOp, mhlo::MaxOp> &&
             isMinValueAttribute(initValue)) {
    return true;
  } else if (std::is_same_v<RegionOp, mhlo::MinOp> &&
             isMaxValueAttribute(initValue)) {
    return true;
  } else if (std::is_same_v<RegionOp, mhlo::OrOp> &&
             isZeroAttribute(initValue)) {
    return true;
  } else if (std::is_same_v<RegionOp, mhlo::MulOp> &&
             isSplatElementsAttribute(cast<DenseIntOrFPElementsAttr>(initValue),
                                      1, 1.0)) {
    return true;
  } else {
    return false;
  }
}

// note: if want to add more instance, need to confirm it is a regular
// reduce/reduce_window.
template bool
    mlir::isRegularReduceOp<mhlo::AddOp, mhlo::ReduceOp>(mhlo::ReduceOp);
template bool
    mlir::isRegularReduceOp<mhlo::MaxOp, mhlo::ReduceOp>(mhlo::ReduceOp);
template bool
    mlir::isRegularReduceOp<mhlo::MinOp, mhlo::ReduceOp>(mhlo::ReduceOp);
template bool
    mlir::isRegularReduceOp<mhlo::OrOp, mhlo::ReduceOp>(mhlo::ReduceOp);
template bool
    mlir::isRegularReduceOp<mhlo::MulOp, mhlo::ReduceOp>(mhlo::ReduceOp);
template bool mlir::isRegularReduceOp<mhlo::AddOp, mhlo::ReduceWindowOp>(
    mhlo::ReduceWindowOp);
template bool mlir::isRegularReduceOp<mhlo::MaxOp, mhlo::ReduceWindowOp>(
    mhlo::ReduceWindowOp);

// Return true if slice region is continuous
bool mlir::isSliceContinuousSubview(mhlo::SliceOp op) {
  auto type = cast<RankedTensorType>(op.getOperand().getType());
  if (!type.hasStaticShape()) {
    return false;
  }
  if (!isSplatValue(op.getStrides(), 1)) {
    return false;
  }

  // find highest non one dimension
  std::optional<int64_t> leadingNonOneDimensionIndex;
  for (int64_t i = 0; i < type.getRank(); i++) {
    if (type.getDimSize(i) != 1) {
      leadingNonOneDimensionIndex = i;
      break;
    }
  }
  if (!leadingNonOneDimensionIndex.has_value()) {
    return false;
  }

  for (int64_t i = 0; i < type.getRank(); i++) {
    if (i != leadingNonOneDimensionIndex.value()) {
      if (op.getStartIndices().getValues<int64_t>()[i] != 0 ||
          op.getLimitIndices().getValues<int64_t>()[i] != type.getDimSize(i)) {
        return false;
      }
    }
  }
  return true;
}

// return cumsum's index, return nullopt if not a cumsum op
std::optional<int64_t> mlir::getCumsumIndex(mhlo::ReduceWindowOp op) {
  auto base_dilations = op.getBaseDilationsAttr();
  if (base_dilations && !isSplatValue(base_dilations, 1)) {
    return std::nullopt;
  }
  auto window_dilations = op.getWindowDilationsAttr();
  if (window_dilations && !isSplatValue(window_dilations, 1)) {
    return std::nullopt;
  }
  auto strides = op.getWindowStridesAttr();
  if (strides && !isSplatValue(strides, 1)) {
    return std::nullopt;
  }

  SmallVector<int64_t> window_dimensions = SmallVector<int64_t>(
      op.getWindowDimensions().getValues<int64_t>().begin(),
      op.getWindowDimensions().getValues<int64_t>().end());
  if (!op.getPadding().has_value()) {
    return std::nullopt;
  }
  SmallVector<int64_t> padding =
      SmallVector<int64_t>(op.getPaddingAttr().getValues<int64_t>().begin(),
                           op.getPaddingAttr().getValues<int64_t>().end());

  auto inputShape = cast<ShapedType>(op.getInputs()[0].getType());
  if (!inputShape.hasRank()) {
    return std::nullopt;
  }
  std::optional<int64_t> index;
  for (int64_t i = 0; i < inputShape.getRank(); i++) {
    if (window_dimensions[i] == 1 && padding[i * 2] == 0 &&
        padding[i * 2 + 1] == 0) {
      // not cumsum index
      continue;
    } else if (inputShape.getDimSize(i) == ShapedType::kDynamic) {
      // doesn't handle dynamic dim
      return std::nullopt;
    } else if (window_dimensions[i] == inputShape.getDimSize(i) &&
               padding[i * 2] == inputShape.getDimSize(i) - 1 &&
               padding[i * 2 + 1] == 0) {
      if (!index.has_value()) {
        index = i;
      } else {
        // more than one dim to be cumsumed
        return std::nullopt;
      }
    } else {
      return std::nullopt;
    }
  }
  return index;
}

template <typename OpTy> bool mlir::isValidPoolOrPoolGradLayout(OpTy op) {
  SmallVector<int64_t> window_dimensions;
  size_t rank;
  if (std::is_same_v<OpTy, mhlo::ReduceWindowOp>) {
    auto reduceWindowOp = dyn_cast<mhlo::ReduceWindowOp>(&op);
    auto base_dilations = reduceWindowOp->getBaseDilationsAttr();
    if (base_dilations && !isSplatValue(base_dilations, 1)) {
      return false;
    }
    auto window_dilations = reduceWindowOp->getWindowDilationsAttr();
    if (window_dilations && !isSplatValue(window_dilations, 1)) {
      return false;
    }
    window_dimensions =
        SmallVector<int64_t>(reduceWindowOp->getWindowDimensions()
                                 .template getValues<int64_t>()
                                 .begin(),
                             reduceWindowOp->getWindowDimensions()
                                 .template getValues<int64_t>()
                                 .end());
    rank = window_dimensions.size();
    if (rank != 3 && rank != 4 && rank != 5) {
      return false;
    }
  } else if (std::is_same_v<OpTy, mhlo::SelectAndScatterOp>) {
    auto selectAndScatterOp = dyn_cast<mhlo::SelectAndScatterOp>(&op);
    if (auto window_dimensions_ =
            selectAndScatterOp->getWindowDimensionsAttr()) {
      window_dimensions = SmallVector<int64_t>(
          window_dimensions_.template getValues<int64_t>().begin(),
          window_dimensions_.template getValues<int64_t>().end());
    } else {
      return false;
    }
    rank = window_dimensions.size();
    if (rank != 4 && rank != 5) {
      return false;
    }
  } else {
    assert(false && "The operation type is unexpected\n");
  }
  if ((window_dimensions[0] != 1) ||
      (window_dimensions[1] != 1 && window_dimensions[rank - 1] != 1)) {
    return false;
  }

  if (auto window_strides = op.getWindowStridesAttr()) {
    SmallVector<int64_t> strides(
        window_strides.template getValues<int64_t>().begin(),
        window_strides.template getValues<int64_t>().end());
    if ((strides[0] != 1) || (strides[1] != 1 && strides[rank - 1] != 1)) {
      return false;
    }
  }
  if (auto padding_ = op.getPaddingAttr()) {
    SmallVector<int64_t> padding(padding_.template getValues<int64_t>().begin(),
                                 padding_.template getValues<int64_t>().end());
    if (padding[0] != 0 || padding[1] != 0 ||
        ((padding[2] != 0 || padding[3] != 0) &&
         (padding[2 * rank - 1 != 0] || padding[2 * rank - 2] != 0))) {
      return false;
    }
  }
  return true;
}

template bool mlir::isValidPoolOrPoolGradLayout(mhlo::ReduceWindowOp);
template bool mlir::isValidPoolOrPoolGradLayout(mhlo::SelectAndScatterOp);

namespace {
byteir::NamedLayout
parsePoolLayout(size_t rank, const SmallVector<int64_t> &window_dimensions,
                const SmallVector<int64_t> &strides,
                const SmallVector<int64_t> &padding) {
  byteir::NamedLayout layout = byteir::NamedLayout::UNKNOWN;
  if (window_dimensions[0] == 1 && window_dimensions[rank - 1] == 1 &&
      strides[0] == 1 && strides[rank - 1] == 1 && padding[0] == 0 &&
      padding[1] == 0 && padding[2 * rank - 2] == 0 &&
      padding[2 * rank - 1] == 0) {
    if (rank == 4) {
      layout = byteir::NamedLayout::NHWC;
    } else if (rank == 5) {
      layout = byteir::NamedLayout::NDHWC;
    }
  } else if (window_dimensions[0] == 1 && window_dimensions[1] == 1 &&
             strides[0] == 1 && strides[1] == 1 && padding[0] == 0 &&
             padding[1] == 0 && padding[2] == 0 && padding[3] == 0) {
    if (rank == 3) {
      layout = byteir::NamedLayout::NCW;
    } else if (rank == 4) {
      layout = byteir::NamedLayout::NCHW;
    } else if (rank == 5) {
      layout = byteir::NamedLayout::NCDHW;
    }
  }
  return layout;
}
} // namespace

byteir::NamedLayout mlir::getPoolLayout(mlir::mhlo::ReduceWindowOp op) {
  if (!mlir::isValidPoolOrPoolGradLayout(op)) {
    return byteir::NamedLayout::UNKNOWN;
  }
  SmallVector<int64_t> window_dimensions = SmallVector<int64_t>(
      op.getWindowDimensions().getValues<int64_t>().begin(),
      op.getWindowDimensions().getValues<int64_t>().end());
  size_t rank = window_dimensions.size();
  SmallVector<int64_t> strides(rank, 1);
  if (auto strides_ = op.getWindowStridesAttr()) {
    strides = SmallVector<int64_t>(strides_.getValues<int64_t>().begin(),
                                   strides_.getValues<int64_t>().end());
  }
  SmallVector<int64_t> padding(rank * 2, 0);
  if (auto padding_ = op.getPaddingAttr()) {
    padding = SmallVector<int64_t>(padding_.getValues<int64_t>().begin(),
                                   padding_.getValues<int64_t>().end());
  }

  return parsePoolLayout(rank, window_dimensions, strides, padding);
}

byteir::NamedLayout mlir::getPoolGradLayout(mlir::mhlo::SelectAndScatterOp op) {
  if (!mlir::isValidPoolOrPoolGradLayout(op)) {
    return byteir::NamedLayout::UNKNOWN;
  }
  SmallVector<int64_t> window_dimensions;
  if (auto window_dimensions_ = op.getWindowDimensionsAttr()) {
    window_dimensions =
        SmallVector<int64_t>(window_dimensions_.getValues<int64_t>().begin(),
                             window_dimensions_.getValues<int64_t>().end());
  }
  size_t rank = window_dimensions.size();
  SmallVector<int64_t> strides(rank, 1);
  if (auto window_strides = op.getWindowStridesAttr()) {
    strides = SmallVector<int64_t>(window_strides.getValues<int64_t>().begin(),
                                   window_strides.getValues<int64_t>().end());
  }
  SmallVector<int64_t> padding(rank * 2, 0);
  if (auto padding_ = op.getPaddingAttr()) {
    padding = SmallVector<int64_t>(padding_.getValues<int64_t>().begin(),
                                   padding_.getValues<int64_t>().end());
  }
  return parsePoolLayout(rank, window_dimensions, strides, padding);
}

std::tuple<byteir::NamedLayout, byteir::NamedLayout, byteir::NamedLayout>
mlir::getConvLayout(mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers) {
  byteir::NamedLayout input_layout = byteir::NamedLayout::UNKNOWN;
  auto input_batch_dimension = dimension_numbers.getInputBatchDimension();
  auto input_feature_dimension = dimension_numbers.getInputFeatureDimension();
  auto input_spatial_dimensions = dimension_numbers.getInputSpatialDimensions();
  if (input_spatial_dimensions.size() == 1) {
    if (input_batch_dimension == 0 && input_feature_dimension == 1) {
      input_layout = byteir::NamedLayout::NCW;
    } else {
      input_layout = byteir::NamedLayout::UNKNOWN;
    }
  } else if (input_spatial_dimensions.size() == 2) {
    if (input_batch_dimension == 0 && input_feature_dimension == 1) {
      input_layout = byteir::NamedLayout::NCHW;
    } else if (input_batch_dimension == 0 && input_feature_dimension == 3) {
      input_layout = byteir::NamedLayout::NHWC;
    } else {
      input_layout = byteir::NamedLayout::UNKNOWN;
    }
  } else if (input_spatial_dimensions.size() == 3) {
    if (input_batch_dimension == 0 && input_feature_dimension == 1) {
      input_layout = byteir::NamedLayout::NCDHW;
    } else if (input_batch_dimension == 0 && input_feature_dimension == 4) {
      input_layout = byteir::NamedLayout::NDHWC;
    } else {
      input_layout = byteir::NamedLayout::UNKNOWN;
    }
  } else {
    input_layout = byteir::NamedLayout::UNKNOWN;
  }

  byteir::NamedLayout output_layout;
  auto output_batch_dimension = dimension_numbers.getOutputBatchDimension();
  auto output_feature_dimension = dimension_numbers.getOutputFeatureDimension();
  auto output_spatial_dimensions =
      dimension_numbers.getOutputSpatialDimensions();
  if (output_spatial_dimensions.size() == 1) {
    if (output_batch_dimension == 0 && output_feature_dimension == 1) {
      output_layout = byteir::NamedLayout::NCW;
    } else {
      output_layout = byteir::NamedLayout::UNKNOWN;
    }
  } else if (output_spatial_dimensions.size() == 2) {
    if (output_batch_dimension == 0 && output_feature_dimension == 1) {
      output_layout = byteir::NamedLayout::NCHW;
    } else if (output_batch_dimension == 0 && output_feature_dimension == 3) {
      output_layout = byteir::NamedLayout::NHWC;
    } else {
      output_layout = byteir::NamedLayout::UNKNOWN;
    }
  } else if (output_spatial_dimensions.size() == 3) {
    if (output_batch_dimension == 0 && output_feature_dimension == 1) {
      output_layout = byteir::NamedLayout::NCDHW;
    } else if (output_batch_dimension == 0 && output_feature_dimension == 4) {
      output_layout = byteir::NamedLayout::NDHWC;
    } else {
      output_layout = byteir::NamedLayout::UNKNOWN;
    }
  } else {
    output_layout = byteir::NamedLayout::UNKNOWN;
  }

  byteir::NamedLayout kernel_layout;
  auto kernel_input_feature_dimension =
      dimension_numbers.getKernelInputFeatureDimension();
  auto kernel_output_feature_dimension =
      dimension_numbers.getKernelOutputFeatureDimension();
  auto kernel_spatial_dimensions =
      dimension_numbers.getKernelSpatialDimensions();
  if (kernel_spatial_dimensions.size() == 1) {
    if (kernel_input_feature_dimension == 1 &&
        kernel_output_feature_dimension == 0) {
      kernel_layout = byteir::NamedLayout::NCW;
    } else {
      kernel_layout = byteir::NamedLayout::UNKNOWN;
    }
  } else if (kernel_spatial_dimensions.size() == 2) {
    if (kernel_input_feature_dimension == 1 &&
        kernel_output_feature_dimension == 0) {
      kernel_layout = byteir::NamedLayout::NCHW;
    } else if (kernel_input_feature_dimension == 3 &&
               kernel_output_feature_dimension == 0) {
      kernel_layout = byteir::NamedLayout::NHWC;
    } else if (kernel_input_feature_dimension == 2 &&
               kernel_output_feature_dimension == 3) {
      kernel_layout = byteir::NamedLayout::HWCN;
    } else {
      kernel_layout = byteir::NamedLayout::UNKNOWN;
    }
  } else if (kernel_spatial_dimensions.size() == 3) {
    if (kernel_input_feature_dimension == 1 &&
        kernel_output_feature_dimension == 0) {
      kernel_layout = byteir::NamedLayout::NCDHW;
    } else if (kernel_input_feature_dimension == 4 &&
               kernel_output_feature_dimension == 0) {
      kernel_layout = byteir::NamedLayout::NDHWC;
    } else if (kernel_input_feature_dimension == 3 &&
               kernel_output_feature_dimension == 4) {
      kernel_layout = byteir::NamedLayout::DHWCN;
    } else {
      kernel_layout = byteir::NamedLayout::UNKNOWN;
    }
  } else {
    kernel_layout = byteir::NamedLayout::UNKNOWN;
  }

  return std::make_tuple(input_layout, kernel_layout, output_layout);
}

template <typename T>
void mlir::handleConvAttribute(NamedAttrList &attrs, T conv_op,
                               OpBuilder &rewriter) {
  auto dimension_numbers = conv_op.getDimensionNumbers();
  auto conv_layout = mlir::getConvLayout(dimension_numbers);

  auto input_layout = std::get<0>(conv_layout);
  auto kernel_layout = std::get<1>(conv_layout);
  auto output_layout = std::get<2>(conv_layout);
  assert(input_layout != byteir::NamedLayout::UNKNOWN &&
         kernel_layout != byteir::NamedLayout::UNKNOWN &&
         output_layout != byteir::NamedLayout::UNKNOWN);
  assert(input_layout == kernel_layout && input_layout == output_layout);

  attrs.append("input_layout",
               rewriter.getStringAttr(byteir::stringifyEnum(input_layout)));
  attrs.append("output_layout",
               rewriter.getStringAttr(byteir::stringifyEnum(output_layout)));
  attrs.append("kernel_layout",
               rewriter.getStringAttr(byteir::stringifyEnum(kernel_layout)));

  if (conv_op.getWindowStrides().has_value()) {
    attrs.append("window_strides", conv_op.getWindowStridesAttr());
  }
  if (conv_op.getPadding().has_value()) {
    attrs.append("padding", conv_op.getPaddingAttr());
  }
  if (conv_op.getLhsDilation().has_value()) {
    attrs.append("lhs_dilation", conv_op.getLhsDilationAttr());
  }
  if (conv_op.getRhsDilation().has_value()) {
    attrs.append("rhs_dilation", conv_op.getRhsDilationAttr());
  }
  attrs.append("feature_group_count", conv_op.getFeatureGroupCountAttr());
  attrs.append("batch_group_count", conv_op.getBatchGroupCountAttr());
  if (conv_op.getWindowReversal().has_value()) {
    attrs.append("window_reversal", conv_op.getWindowReversalAttr());
  }
}

// instantiate
template void mlir::handleConvAttribute<mhlo::ConvolutionOp>(
    NamedAttrList &, mhlo::ConvolutionOp, OpBuilder &);

namespace {

// this function copied from mlir-hlo/lib/Dialect/mhlo/IR/hlo_ops.cc
DenseElementsAttr reshape(DenseElementsAttr attr, ShapedType newType) {
  // TODO(b/232866626): DenseElementsAttr::reshape is broken for bool splats.
  // Once that ticket is fixed, we can remove this conditional.
  if (attr.isSplat() && newType.getElementType().isInteger(/*width=*/1)) {
    auto splatValue = attr.getValues<bool>()[0];
    return DenseElementsAttr::get(newType, {splatValue});
  }
  return attr.reshape(newType);
}

template <typename ValType>
Attribute
createBroadcastedDenseElementsAttrImpl(DenseElementsAttr originAttr,
                                       ShapedType newType,
                                       ArrayRef<int64_t> broadcastDims) {
  SmallVector<ValType> originValues{originAttr.getValues<ValType>().begin(),
                                    originAttr.getValues<ValType>().end()};
  SmallVector<ValType> newValues;
  newValues.reserve(newType.getNumElements());

  auto getStrides = [](ArrayRef<int64_t> shape) {
    SmallVector<int64_t> strides(shape.size(), 1);
    for (int64_t i = strides.size() - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
  };
  SmallVector<int64_t> originStrides =
      getStrides(originAttr.getType().getShape());
  auto outShape = newType.getShape();
  SmallVector<int64_t> outStrides = getStrides(outShape);
  SmallVector<int64_t> dimMapping(newType.getRank(), K_INITIAL);
  for (size_t i = 0; i < broadcastDims.size(); ++i) {
    dimMapping[broadcastDims[i]] = i;
  }

  // return the original index and increment current index by 1.
  auto indexIncrement = [&](SmallVector<int64_t> &curIndex) {
    int64_t originIndex = 0;
    for (size_t i = 0; i < curIndex.size(); ++i) {
      if (dimMapping[i] >= 0) {
        originIndex += originStrides[dimMapping[i]] * curIndex[i];
      }
    }

    for (int64_t i = curIndex.size() - 1; i >= 0; --i) {
      curIndex[i] = (curIndex[i] + 1) % outShape[i];
      if (curIndex[i] != 0)
        break;
    }

    return originIndex;
  };

  SmallVector<int64_t> curIndex(outShape.size(), 0);
  for (int64_t i = 0; i < newType.getNumElements(); ++i) {
    int64_t originIndex = indexIncrement(curIndex);
    newValues.push_back(originValues[originIndex]);
  }
  return DenseElementsAttr::get(newType, newValues);
}

} // namespace

std::optional<Attribute>
mlir::createBroadcastedDenseElementsAttr(DenseElementsAttr originAttr,
                                         ShapedType newType,
                                         ArrayRef<int64_t> broadcastDims) {
  // deduce originAttr's dimension which == 1
  ShapedType valueType = originAttr.getType();
  llvm::SmallVector<int64_t> newBroadcastDims;
  if (llvm::any_of(valueType.getShape(),
                   [](int64_t dim) { return dim == 1; })) {
    llvm::SmallVector<int64_t> newValueShape;
    for (unsigned i = 0, e = valueType.getRank(); i < e; ++i) {
      if (valueType.getDimSize(i) != 1) {
        newValueShape.push_back(valueType.getDimSize(i));
        newBroadcastDims.push_back(broadcastDims[i]);
      }
    }
    auto newValueType =
        RankedTensorType::get(newValueShape, valueType.getElementType());
    originAttr = reshape(originAttr, newValueType);
  } else {
    newBroadcastDims = llvm::to_vector(broadcastDims);
  }

  if (isa<FloatType>(valueType.getElementType())) {
    return createBroadcastedDenseElementsAttrImpl<APFloat>(originAttr, newType,
                                                           newBroadcastDims);
  } else if (isa<IntegerType>(valueType.getElementType())) {
    return createBroadcastedDenseElementsAttrImpl<APInt>(originAttr, newType,
                                                         newBroadcastDims);
  }
  return std::nullopt;
}

namespace {
SmallVector<int64_t> computeStrides(ArrayRef<int64_t> shape) {
  assert(shape.size() > 0);
  SmallVector<int64_t> strides(shape.size(), 0);
  strides[shape.size() - 1] = 1;
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}
} // namespace

std::optional<SmallVector<int64_t>>
mlir::computeReshapeInputOutputRankMapIndex(ShapedType inputType,
                                            ShapedType outputType) {
  if (!inputType.hasStaticShape() || !outputType.hasStaticShape()) {
    return std::nullopt;
  }
  if (inputType.getRank() == 0 || outputType.getRank() == 0) {
    return std::nullopt;
  }
  if (inputType.getRank() >= outputType.getRank()) {
    return std::nullopt;
  }

  auto inputStrides = computeStrides(inputType.getShape());
  auto outputStrides = computeStrides(outputType.getShape());
  SmallVector<int64_t> result;
  int64_t j = 0;
  for (int64_t i = 0; i < inputType.getRank(); i++) {
    while (j < outputType.getRank()) {
      if (inputType.getDimSize(i) == outputType.getDimSize(j) &&
          inputStrides[i] == outputStrides[j]) {
        result.push_back(j);
        j++;
        break;
      }
      j++;
    }
  }
  if (result.size() != static_cast<size_t>(inputType.getRank())) {
    return std::nullopt;
  }
  return result;
}

// compute the index of the reshape's expand dimension
std::optional<int64_t>
mlir::computeReshapeExpandDim(mhlo::ReshapeOp reshapeOp) {
  auto reshapeOperandType = cast<ShapedType>(reshapeOp.getOperand().getType());
  if (!reshapeOperandType.hasStaticShape()) {
    return std::nullopt;
  }
  auto reshapeResultType = cast<ShapedType>(reshapeOp.getResult().getType());

  auto maybeIndex = computeReshapeInputOutputRankMapIndex(reshapeOperandType,
                                                          reshapeResultType);
  // only support using Reshape to expand one dimension
  if ((!maybeIndex.has_value()) ||
      (reshapeOperandType.getRank() != reshapeResultType.getRank() - 1)) {
    return std::nullopt;
  }
  auto index = *maybeIndex;
  for (int64_t i = 0; i < reshapeOperandType.getRank(); i++) {
    if (index[i] != i) {
      return i;
    }
  }
  return reshapeOperandType.getRank();
}

FailureOr<SmallVector<Value>>
mlir::createEmptyTensorForOpResult(OpBuilder &builder, Operation *op) {
  SmallVector<Value> emptyTensors;
  bool resultsHasDynamicShape = false;
  for (auto &&result : op->getResults()) {
    if (auto resType = dyn_cast<ShapedType>(result.getType())) {
      if (resType.hasStaticShape()) {
        auto emptyOp = builder.create<tensor::EmptyOp>(
            op->getLoc(), resType.getShape(), resType.getElementType());
        emptyTensors.emplace_back(emptyOp);
      } else {
        resultsHasDynamicShape = true;
        break;
      }
    }
  }

  if (resultsHasDynamicShape) {
    emptyTensors.clear();
    registerAllMhloReifyReturnTypeShapes();
    SmallVector<Value, 1> reifications;

    if (reifyShapes(builder, op, reifications).failed()) {
      return failure();
    }

    for (auto &&resultAndShape : llvm::zip(op->getResults(), reifications)) {
      SmallVector<Value, 1> dynamicSizes;
      auto resType = cast<ShapedType>(std::get<0>(resultAndShape).getType());
      for (int64_t i = 0; i < resType.getRank(); ++i) {
        if (resType.isDynamicDim(i)) {
          auto dim = builder
                         .create<tensor::ExtractOp>(
                             op->getLoc(), std::get<1>(resultAndShape),
                             ValueRange{builder.create<arith::ConstantIndexOp>(
                                 op->getLoc(), static_cast<int64_t>(i))})
                         .getResult();
          dynamicSizes.emplace_back(dim);
        }
      }
      auto emptyOp =
          builder.create<tensor::EmptyOp>(op->getLoc(), resType, dynamicSizes);
      emptyTensors.emplace_back(emptyOp);
    }
  }
  return emptyTensors;
}
