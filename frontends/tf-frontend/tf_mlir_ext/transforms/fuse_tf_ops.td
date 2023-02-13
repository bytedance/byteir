//===- fuse_tf_ops.td ------------------------------------*--- tablegen -*-===//
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

#ifndef FUSE_TF_OPS_PATTERN
#define FUSE_TF_OPS_PATTERN

include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"
include "mhlo/IR/hlo_ops.td"

//===----------------------------------------------------------------------===//
// Sigmoid Pattern
//===----------------------------------------------------------------------===//

def SplatValueOne : Constraint<CPred<"isSplatValue($0.dyn_cast<DenseFPElementsAttr>(), 1.0)">, "value attr is not splat value of 1.0">;

def FuseSigmoid : Pat<
  (TF_ReciprocalOp
    (TF_AddV2Op
      (TF_ExpOp
        (TF_NegOp $input)
      ),
      (TF_ConstOp $one)
    )
  ),
  (TF_SigmoidOp $input),
  [(SplatValueOne $one)]>;



#endif // FUSE_TF_OPS_PATTERN
