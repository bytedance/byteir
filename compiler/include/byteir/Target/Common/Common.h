//===- Common.h -----------------------------------------------------------===//
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

#ifndef BYTEIR_TARGET_COMMON_COMMON_H
#define BYTEIR_TARGET_COMMON_COMMON_H

namespace mlir {

enum OptLevel { O0 = 0, O1, O2, O3 };

} // namespace mlir

#endif // BYTEIR_TARGET_COMMON_COMMON_H