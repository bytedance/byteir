//===- Convolution.cpp ----------------------------------------*--- C++ -*-===//
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
///
/// \file
/// FIXME: remove if upstream have implemented shape inference
///
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/DynamicShapeOpRegister/Register.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "dynamic-shape-op-register"

using namespace mlir;

namespace {
/// NOTE: following code is copied from hlo_ops.cc for shape inference

// Convert a 1D dense int64 attribute to a list of values.
SmallVector<int64_t>
convertDenseIntAttr(std::optional<mlir::DenseIntElementsAttr> optionalAttr) {
  if (!optionalAttr.has_value())
    return SmallVector<int64_t>{};

  mlir::DenseIntElementsAttr attr = *optionalAttr;
  auto values = attr.getValues<int64_t>();
  return {values.begin(), values.end()};
}

// Convert a 1D or Nx2 dense int64 attribute to a list of tuples.
FailureOr<SmallVector<std::pair<int64_t, int64_t>>>
convertNx2Attribute(std::optional<mlir::DenseIntElementsAttr> optionalAttr,
                    Location loc) {
  if (!optionalAttr.has_value())
    return SmallVector<std::pair<int64_t, int64_t>>{};
  mlir::DenseIntElementsAttr attr = *optionalAttr;

  auto attrType = attr.getType().cast<RankedTensorType>(); // ensured by ODS.
  if (attrType.getRank() > 1) {
    if (attrType.getRank() != 2 || attrType.getShape()[1] != 2)
      return (mlir::emitError(loc) << "expects the shape of padding-attribute "
                                      "to be {N, 2}, but got {"
                                   << attrType.getShape() << "}.",
              failure());
  } else {
    // Padding values can be provided as a 1D vector as well.
    if (attr.getValues<int64_t>().size() % 2 != 0)
      return (mlir::emitError(loc)
                  << "expects the padding-entries to have even number of "
                     "elements, but got "
                  << attr.getValues<int64_t>().size() << " elements.",
              failure());
  }

  auto it = attr.getValues<int64_t>().begin();
  SmallVector<std::pair<int64_t, int64_t>> out(attr.getNumElements() / 2);
  for (auto &item : out) {
    int64_t first = *it;
    ++it;
    int64_t second = *it;
    ++it;
    item = {first, second};
  }
  return out;
}

// Check if the dimension size is dynamic.
inline static bool isDynamicDimSize(int64_t val) {
  return val == ShapedType::kDynamic;
}

// If a window with the given bound in some dimension is dilated with the given
// dilation factor in that dimension, then the value returned is the bound for
// the array in that dimension after dilation.
//
// For a 1D array with 3 entries 1, 2, 3, a dilation factor of 2 yields a new
// window with values 1, x, 2, x, 3, where x indicates holes left by the
// dilation. So DilatedBound(3, 2) == 5.
int64_t dilatedBound(int64_t bound, int64_t dilation) {
  assert(bound >= 0 && "The dimension to dialate must be >= 0");
  if (bound == 0)
    return 0;

  // Suppose the array has three entries 123 and the dilation factor is 4. Then
  // the dilated array has 9 entries 1xxx2xxx3. Here, each original entry except
  // the last expands into 4 entries, so that is (bound - 1) * dilation. Then we
  // add 1 to account for the final input element.
  return (bound - 1) * dilation + 1;
}

// Returns the number of valid positions of a window with the given size and
// stride within an array with the given bound. This is the bound of an output
// array with one element per valid position of the window.
//
// For example, for arguments of (bound=5, window_size=2, stride=2), the
// returned value is 2. There are valid positions at offset 0 and offset 2,
// while offset 4 is not valid since the window's last entry would be at 5,
// which is beyond the bound of 5.
int64_t stridedBound(int64_t bound, int64_t windowSize, int64_t stride) {
  assert(windowSize >= 0 && "Expected window size to be >= 0");
  assert(bound >= 0 && "Expected bound to be >= 0");

  if (bound == 0 || windowSize > bound)
    return 0;

  // Without considering stride, the maximum valid offset is bound -
  // window_size. Taking stride into account, the valid offsets then have the
  // form q * stride for q = 0, ..., Q such that q * stride <= bound -
  // window_size. This implies that Q equals floor(bound - window_size /
  // stride). There are Q + 1 valid values of q, yielding the formula below.
  return (bound - windowSize) / stride + 1;
}

// WindowDimension described how the kernel window moves across the base area
// in a particular dimension.
// Describes the windowing in an operation such as convolution.
// The window is moved across a base area and for each position of the
// window a computation is performed. The field below describes the
// window and the movement of the window across a base area.
struct WindowDimension {
  int64_t size = 0;
  int64_t stride = 1;
  int64_t paddingLow = 0;
  int64_t paddingHigh = 0;
  int64_t windowDilation = 1;
  int64_t baseDilation = 1;
  bool windowReversal = false;
};

// Verifies various properties of window-attributes (viz., stride, padding,
// lhs_dilation and rhs_dilation) and collects all the window-attributes for
// each kernel spatial dimensions.
FailureOr<SmallVector<WindowDimension>>
verifyWindowAttributesAndInferWindowDimensions(
    ArrayRef<int64_t> windowDimensions, ArrayRef<int64_t> windowStrides,
    ArrayRef<std::pair<int64_t, int64_t>> padding,
    ArrayRef<int64_t> lhsDilation, ArrayRef<int64_t> rhsDilation,
    Location loc) {
  const auto verifySize = [&](const size_t attrSize,
                              StringRef attrName) -> LogicalResult {
    if (attrSize == 0 || attrSize == windowDimensions.size())
      return success();
    return mlir::emitError(loc)
           << "expects " << attrName
           << " to have same dimension-size as size of "
              "window dimensions "
              "("
           << windowDimensions.size() << "), but got: " << attrSize << ".";
  };

  if (failed(verifySize(windowStrides.size(), "window-strides")))
    return failure();
  if (failed(verifySize(lhsDilation.size(), "base-dilation factors")))
    return failure();
  if (failed(verifySize(rhsDilation.size(), "window-dilation factors")))
    return failure();
  if (failed(verifySize(padding.size(), "padding-entries")))
    return failure();

  SmallVector<WindowDimension> window(windowDimensions.size());
  for (size_t i = 0; i < windowDimensions.size(); i++) {
    WindowDimension &dim = window[i];

    dim.size = windowDimensions[i];
    if (!isDynamicDimSize(dim.size) && dim.size <= 0)
      return (mlir::emitError(loc)
                  << "expects window to have positive value for " << i
                  << "-th window dimension, but got " << dim.size << ".",
              failure());

    if (!windowStrides.empty())
      dim.stride = windowStrides[i];
    if (dim.stride <= 0)
      return (mlir::emitError(loc)
                  << "expects window to have positive stride for " << i
                  << "-th window dimension, but got " << dim.stride << ".",
              failure());

    if (!lhsDilation.empty())
      dim.baseDilation = lhsDilation[i];
    if (dim.baseDilation <= 0)
      return (mlir::emitError(loc) << "expects window to have positive base "
                                      "dilation factor for "
                                   << i << "-th window dimension, but got "
                                   << dim.baseDilation << ".",
              failure());

    if (!rhsDilation.empty())
      dim.windowDilation = rhsDilation[i];
    if (dim.windowDilation <= 0)
      return (mlir::emitError(loc) << "expects window to have positive window "
                                      "dilation factor for "
                                   << i << "-th window dimension, but got "
                                   << dim.windowDilation << ".",
              failure());

    if (!padding.empty()) {
      dim.paddingLow = padding[i].first;
      dim.paddingHigh = padding[i].second;
    }
  }

  return window;
}

// Infer the shape of the output window.
//  Foreach dimension d,
//    output-window-shape[d] =
//            stridedBound(padding_low + dilatedBound(base_shape[d]) +
//            padding_high,
//                         dilatedBound(window_shape[d]))
//      where (padding_low, padding_high) is the padding-pair for d.
SmallVector<int64_t>
inferWindowOutputShape(const ArrayRef<int64_t> baseShape,
                       const ArrayRef<WindowDimension> window) {
  assert(baseShape.size() == window.size() &&
         "Size of window dimensions must match the size of base shape.");

  SmallVector<int64_t> outputDimensions(window.size());
  for (size_t i = 0; i < window.size(); ++i) {
    if (isDynamicDimSize(baseShape[i]) || isDynamicDimSize(window[i].size)) {
      outputDimensions[i] = ShapedType::kDynamic;
    } else {
      const auto &dim = window[i];

      const int64_t dilatedBase = dilatedBound(baseShape[i], dim.baseDilation);
      const int64_t paddedDilatedBase =
          dim.paddingLow + dilatedBase + dim.paddingHigh;
      const int64_t dilatedWindow = dilatedBound(dim.size, dim.windowDilation);

      outputDimensions[i] =
          stridedBound(paddedDilatedBase, dilatedWindow, dim.stride);
    }
  }

  return outputDimensions;
}
} // namespace

void mlir::registerConvolutionInferReturnTypeComponents() {
  static InferReturnTypeComponentsRegistration shapeRegister(
      mhlo::ConvolutionOp::getOperationName(),
      [](MLIRContext *context, std::optional<Location> location,
         ValueShapeRange operands, DictionaryAttr attrs, RegionRange regions,
         SmallVectorImpl<ShapedTypeComponents> &inferredReturnTypes) {
        mhlo::ConvolutionOp::Adaptor adaptor(operands, attrs, {}, regions);
        Location loc = location.value_or(UnknownLoc::get(context));

        if (failed(adaptor.verify(loc))) {
          LLVM_DEBUG(llvm::dbgs() << loc << ": conv op verify failed\n");
          return failure();
        }

        auto lhsType = adaptor.getLhs().getType().dyn_cast<RankedTensorType>();
        auto rhsType = adaptor.getRhs().getType().dyn_cast<RankedTensorType>();
        if (!lhsType || !rhsType) {
          LLVM_DEBUG(llvm::dbgs() << loc << ": operands type missing\n");
          return failure();
        }

        auto kernelSpatialDimensions =
            adaptor.getDimensionNumbers().getKernelSpatialDimensions();
        SmallVector<int64_t> windowDimensions(kernelSpatialDimensions.size());
        for (size_t i = 0; i < windowDimensions.size(); i++)
          windowDimensions[i] = rhsType.getShape()[kernelSpatialDimensions[i]];

        auto paddingOrErr = convertNx2Attribute(adaptor.getPadding(), loc);
        if (failed(paddingOrErr)) {
          LLVM_DEBUG(llvm::dbgs() << "parse padding failed\n");
          return failure();
        }
        SmallVector<std::pair<int64_t, int64_t>> padding = *paddingOrErr;

        // adaptor.getDimensionNumbers() return failure();
        auto windowOrErr = verifyWindowAttributesAndInferWindowDimensions(
            windowDimensions, convertDenseIntAttr(adaptor.getWindowStrides()),
            padding, convertDenseIntAttr(adaptor.getLhsDilation()),
            convertDenseIntAttr(adaptor.getRhsDilation()), loc);
        if (failed(windowOrErr))
          return failure();

        SmallVector<WindowDimension> window = *windowOrErr;

        const int64_t dataRank = lhsType.getRank();
        SmallVector<int64_t> outputDimensions(dataRank, ShapedType::kDynamic);

        SmallVector<int64_t> lhsShape;
        SmallVector<int64_t> rhsShape;
        operands.getShape(0).getDims(lhsShape);
        operands.getShape(1).getDims(rhsShape);

        // Infer the output spatial dimensions.
        auto inputSpatialDims =
            adaptor.getDimensionNumbers().getInputSpatialDimensions();
        auto numSpatialDims = inputSpatialDims.size();
        SmallVector<int64_t> inputSpatialDimVals(numSpatialDims);

        for (size_t i = 0; i < numSpatialDims; ++i)
          inputSpatialDimVals[i] = lhsShape[inputSpatialDims[i]];

        auto windowOutputShape =
            inferWindowOutputShape(inputSpatialDimVals, *windowOrErr);

        for (int i = 0, e = window.size(); i < e; ++i)
          outputDimensions[adaptor.getDimensionNumbers()
                               .getOutputSpatialDimensions()[i]] =
              windowOutputShape[i];

        // Infer the output-batch-dimension and output-feature-dimension.
        const int64_t inputBatch =
            lhsShape[adaptor.getDimensionNumbers().getInputBatchDimension()];
        const int64_t kernelOutputFeatures =
            rhsShape[adaptor.getDimensionNumbers()
                         .getKernelOutputFeatureDimension()];

        outputDimensions[adaptor.getDimensionNumbers()
                             .getOutputBatchDimension()] =
            isDynamicDimSize(inputBatch)
                ? ShapedType::kDynamic
                : inputBatch / adaptor.getBatchGroupCount();
        outputDimensions[adaptor.getDimensionNumbers()
                             .getOutputFeatureDimension()] =
            kernelOutputFeatures;

        Type outputElemType = lhsType.getElementType();
        inferredReturnTypes.emplace_back(outputDimensions, outputElemType);
        return success();
      });
}
