//===- models.h -----------------------------------------------*--- C++ -*-===//
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

#pragma once

#include "brt/core/framework/dtype.h"
#include "brt/core/ir/builder.h"
#include <string>
#include <vector>

namespace brt {
namespace test {

const void *CreateAddOp2(brt::ir::ByREBuilder &byre_builder,
                         const std::string &space);

const void *CreateAddWeight(brt::ir::ByREBuilder &byre_builder,
                            const std::string &space);

const void *CreateCopyOp(brt::ir::ByREBuilder &byre_builder,
                         const std::string &src_space,
                         const std::string &dst_space);

const void *CreateCustom(brt::ir::ByREBuilder &byre_builder,
                         const std::string &space);

const void *CreateUnknown(brt::ir::ByREBuilder &byre_builder,
                          const std::string &space);

const void *CreateMatmul(brt::ir::ByREBuilder &byre_builder, DTypeEnum dataType,
                         const std::string &space, int64_t m, int64_t n,
                         int64_t k, int64_t lhs_contracting_dimension = 1,
                         int64_t rhs_contracting_dimension = 0,
                         bool output_transpose = false,
                         bool compute_on_fp16 = false);

const void *CreateMatmul2(brt::ir::ByREBuilder &byre_builder,
                          const std::string &space);

const void *CreateBatchMatmul(brt::ir::ByREBuilder &byre_builder,
                              DTypeEnum dataType, const std::string &space,
                              llvm::ArrayRef<int64_t> b, int64_t m, int64_t n,
                              int64_t k, int64_t lhs_contracting_dimension,
                              int64_t rhs_contracting_dimension);

const void *CreateConv(brt::ir::ByREBuilder &byre_builder, const std::string op,
                       DTypeEnum dataType, const std::string &space, int64_t N,
                       int64_t iC, int64_t iH, int64_t iW, int64_t oC,
                       int64_t kH, int64_t kW, const std::string &layout,
                       int64_t strideH, int64_t strideW, int64_t paddingH,
                       int64_t paddingW, int64_t dilateH, int64_t dilateW);

const void *CreatePoolMax(brt::ir::ByREBuilder &byre_builder,
                          DTypeEnum dataType, const std::string &space,
                          std::vector<int64_t> &shape_input,
                          std::vector<int64_t> &shape_output,
                          std::vector<int64_t> &padding,
                          std::vector<int64_t> &window_dimensions,
                          std::vector<int64_t> &window_strides);

const void *CreatePoolMaxGrad(brt::ir::ByREBuilder &byre_builder,
                              DTypeEnum dataType, const std::string &space,
                              std::vector<int64_t> &shape_x,
                              std::vector<int64_t> &shape_y,
                              std::vector<int64_t> &padding,
                              std::vector<int64_t> &window_dimensions,
                              std::vector<int64_t> &window_strides);

const void *CreateBatchNormTraining(brt::ir::ByREBuilder &byre_builder,
                                    DTypeEnum dataType,
                                    const std::string &space,
                                    std::vector<int64_t> &shape_input,
                                    int64_t feature_index, float epsilon);

const void *CreateBatchNormGrad(brt::ir::ByREBuilder &byre_builder,
                                DTypeEnum dataType, const std::string &space,
                                std::vector<int64_t> &shape_input,
                                int64_t feature_index, float epsilon);

const void *CreateIndexPut(brt::ir::ByREBuilder &byre_builder,
                           const std::string &space,
                           std::vector<int64_t> src_shape, size_t dim,
                           std::vector<int64_t> idx_shape);

const void *CreateIndexSelect(brt::ir::ByREBuilder &byre_builder,
                              const std::string &space,
                              std::vector<int64_t> src_shape, size_t dim,
                              std::vector<int64_t> idx_shape,
                              bool is_ui32_index);

const void *CreateReduction(brt::ir::ByREBuilder &byre_builder,
                            const std::string &space,
                            std::vector<int64_t> src_shape,
                            std::vector<int64_t> dimensions,
                            std::string reduce_op);

const void *CreateTopK(brt::ir::ByREBuilder &byre_builder, DTypeEnum dataType,
                       DTypeEnum indexType, std::vector<int64_t> src_shape,
                       int64_t k, std::vector<int64_t> axis_vec, bool sorted);

const void *CreateTranspose(brt::ir::ByREBuilder &byre_builder,
                            DTypeEnum dataType, const std::string &space,
                            std::vector<int64_t> &shape_input,
                            std::vector<int64_t> &shape_output,
                            std::vector<int64_t> &permutation);

const void *CreateTypecvt(brt::ir::ByREBuilder &byre_builder,
                          DTypeEnum src_dtype, DTypeEnum dst_dtype,
                          const std::vector<int64_t> &shape);

const void *CreateAliasThenIndexPut(brt::ir::ByREBuilder &byre_builder,
                                    const std::string &space,
                                    std::vector<int64_t> data_src_shape,
                                    int64_t idx_src_len, int64_t idx_dst_len,
                                    int32_t idx_offset);

const void *CreateRepeat(brt::ir::ByREBuilder &byre_builder, DTypeEnum dataType,
                         DTypeEnum timesType, std::vector<int64_t> data_shape,
                         std::vector<int64_t> times_shape,
                         std::vector<int64_t> output_shape);

// always cuda
const void *CreatePTXAddOp(brt::ir::ByREBuilder &byre_builder);

const void *CreateTFWhereOp(brt::ir::ByREBuilder &byre_builder,
                            DTypeEnum input_dtype,
                            const std::vector<int64_t> &shape);

const void *CreateTFSelectOp(brt::ir::ByREBuilder &byre_builder,
                             DTypeEnum dtype,
                             const std::vector<int64_t> &cond_shape,
                             const std::vector<int64_t> &input_shape);

const void *CreateTFStringToNumberOp(brt::ir::ByREBuilder &byre_builder,
                                     DTypeEnum InType, DTypeEnum OutType,
                                     const std::vector<int64_t> &input_shape);

const void *
CreateWithEntryAttrs(brt::ir::ByREBuilder &byre_builder, DTypeEnum input_dtype,
                     const std::vector<int64_t> &shape,
                     const std::vector<std::string> &inputs,
                     const std::vector<std::string> &outputs,
                     const std::vector<std::string> &original_inputs);
} // namespace test
} // namespace brt
