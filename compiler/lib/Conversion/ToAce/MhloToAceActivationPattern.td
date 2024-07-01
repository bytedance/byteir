//===- MhloToAceActivationPattern.td --------------------------------------===//
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

#ifndef BYTEIR_CONVERSION_TOACE_MHLO_TO_ACE_ACTIVATION_PATTERN
#define BYTEIR_CONVERSION_TOACE_MHLO_TO_ACE_ACTIVATION_PATTERN


include "byteir/Dialect/Ace/AceOps.td"
include "mhlo/IR/hlo_ops.td"

def CreateAceActivateOp : NativeCodeCall<"$_builder.create<mlir::ace::ActivateOp>($_loc, $0.getType(), $1, $2, $3)">;
def AttrIsZero : Constraint<CPred<"mlir::isZeroAttribute($_self)">>;
def FPDenseSixValue : Constraint<CPred<"mlir::isSplatCloseToValue(dyn_cast<DenseFPElementsAttr>($_self), 6.0)">>;
def FPDenseSplatValue : Constraint<CPred<"dyn_cast<DenseFPElementsAttr>($_self) && dyn_cast<DenseFPElementsAttr>($_self).isSplat()">>;


def MhloToAceReluPattern : Pat<
  (MHLO_MaxOp
    $input,
    (MHLO_ConstantOp
      $value
    )
  ),
  (Ace_ActivateOp $input, (NativeCodeCall<"$_builder.getStringAttr(\"relu\")">)),
  [(AttrIsZero:$value)]>;

def MhloToAceRelu6Pattern : Pat<
  (MHLO_ClampOp
    (MHLO_ConstantOp
      $value1
    ),
    $input,
    (MHLO_ConstantOp
      $value2
    )
  ),
  (Ace_ActivateOp $input, (NativeCodeCall<"$_builder.getStringAttr(\"relu6\")">)),
  [(AttrIsZero:$value1), (FPDenseSixValue:$value2)]>;

def MhloToAceLeakyReluPattern : Pat<
  (MHLO_SelectOp:$output
    (MHLO_CompareOp
      $input,
      (MHLO_ConstantOp $value1),
      // HLO_COMPARISON_DIRECTION_GT, TODO: fix this at mlir-tblgen
      ConstantAttr<MHLO_ComparisonDirectionAttr, "mhlo::ComparisonDirection::GT">:$comparion_direction,
      $_
    ),
    $input,
    (MHLO_MulOp
      $input,
      (MHLO_ConstantOp $value2)
    )
  ),
  (CreateAceActivateOp $output, $input, (NativeCodeCall<"$_builder.getStringAttr(\"leaky_relu\")">),
    (NativeCodeCall<"{$_builder.getNamedAttr(\"alpha\", cast<DenseFPElementsAttr>(value2).getSplatValue<FloatAttr>())}">)),
  [(AttrIsZero:$value1), (FPDenseSplatValue:$value2)]>;

#endif // BYTEIR_CONVERSION_TOACE_MHLO_TO_ACE_ACTIVATION_PATTERN