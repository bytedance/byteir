//===- alias_test.cc ------------------------------------------*--- C++ -*-===//
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

#include "brt/backends/cuda/device/cuda_allocator.h"
#include "brt/backends/cuda/providers/default/cuda_provider.h"
#include "brt/core/common/status.h"
#include "brt/core/session/request_context.h"
#include "brt/core/session/session.h"
#include "brt/test/common/cuda/util.h"
#include "brt/test/common/models.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cstdlib>
#include <cuda_runtime.h>
#include <memory>
#include <string>

using namespace brt;
using namespace brt::common;
using namespace brt::ir;
using namespace brt::test;

namespace {
size_t getNumElementsOfShape(const std::vector<int64_t> &shape) {
  size_t ret = 1;
  for (auto &&i : shape) {
    ret *= i;
  }
  return ret;
}

void CheckAliasThenIndexPutFisrtDimCUDAValue(
    void *d_input, void *d_index, void *d_update, void *d_output,
    const std::vector<int64_t> &inout_shape, int64_t idx_src_len,
    int64_t idx_dst_len, int32_t idx_offset, float eps) {

  size_t update_size = idx_dst_len;
  int feature_size = 1;
  for (size_t i = 1; i < inout_shape.size(); ++i) {
    feature_size *= inout_shape[i];
  }

  update_size *= feature_size;

  size_t inout_size = getNumElementsOfShape(inout_shape);
  size_t inout_size_in_bytes = inout_size * sizeof(float);

  size_t index_size_in_bytes = idx_src_len * sizeof(int64_t);
  size_t update_size_in_bytes = update_size * sizeof(float);

  float *h_input = (float *)malloc(inout_size_in_bytes);
  int64_t *h_index = (int64_t *)malloc(index_size_in_bytes);
  float *h_update = (float *)malloc(update_size_in_bytes);
  float *h_output = (float *)malloc(inout_size_in_bytes);

  cudaMemcpy(h_input, d_input, inout_size_in_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_index, d_index, index_size_in_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_update, d_update, update_size_in_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_output, d_output, inout_size_in_bytes, cudaMemcpyDeviceToHost);

  std::vector<float> result(inout_size);
  std::copy(h_input, h_input + inout_size, result.begin());

  for (int i = 0; i < idx_dst_len; ++i) {
    int in_offset = i * feature_size;
    int out_offset = h_index[idx_offset + i] * feature_size;
    for (int j = 0; j < feature_size; ++j) {
      int out_index = out_offset + j;
      int in_index = in_offset + j;
      result[out_index] += h_update[in_index];
    }
  }

  for (size_t i = 0; i < inout_size; ++i) {
    EXPECT_NEAR(result[i], h_output[i], eps);
  }

  free(h_input);
  free(h_index);
  free(h_update);
  free(h_output);
}

void CheckIndexPutFirstDim(const std::vector<int64_t> &inout_shape,
                           int64_t idx_src_len, int64_t idx_dst_len,
                           int32_t idx_offset) {
  Session session;
  auto status_allocator = CUDAAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);
  auto status_cuda = DefaultCUDAExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cuda);

  ByREBuilder byre_builder;
  auto status_load = session.LoadFromMemory(
      CreateAliasThenIndexPut(byre_builder, "cuda", inout_shape, idx_src_len,
                              idx_dst_len, idx_offset),
      "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  std::unique_ptr<RequestContext> request;
  auto status_request = session.NewRequestContext(&request);
  BRT_TEST_CHECK_STATUS(status_request);
  size_t update_size = idx_dst_len;
  int feature_size = 1;
  for (size_t i = 1; i < inout_shape.size(); ++i) {
    feature_size *= inout_shape[i];
  }
  update_size *= feature_size;
  std::vector<int64_t> index_shape = {idx_src_len};

  // initiate input
  RandCUDABuffer((float *)request->GetArg(0),
                 getNumElementsOfShape(inout_shape));
  // initiate index
  RandCUDABuffer((int64_t *)request->GetArg(1),
                 getNumElementsOfShape(index_shape), inout_shape[0]);
  // initiate update
  RandCUDABuffer((float *)request->GetArg(2), update_size);

  request->FinishIOBinding();

  auto status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);
  auto status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);

  float eps = 1e-5f;
  CheckAliasThenIndexPutFisrtDimCUDAValue(
      request->GetArg(0), request->GetArg(1), request->GetArg(2),
      request->GetArg(3), inout_shape, idx_src_len, idx_dst_len, idx_offset,
      eps);

  // second run
  RandCUDABuffer((float *)request->GetArg(0),
                 getNumElementsOfShape(inout_shape));
  RandCUDABuffer((int64_t *)request->GetArg(1),
                 getNumElementsOfShape(index_shape), inout_shape[0]);
  RandCUDABuffer((float *)request->GetArg(2), update_size);

  status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);

  status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);

  CheckAliasThenIndexPutFisrtDimCUDAValue(
      request->GetArg(0), request->GetArg(1), request->GetArg(2),
      request->GetArg(3), inout_shape, idx_src_len, idx_dst_len, idx_offset,
      eps);
}

} // namespace

TEST(CUDATestAliasOp, AliasThenIndexPut) {
  CheckIndexPutFirstDim({3, 2}, 6, 3, 0);
  CheckIndexPutFirstDim({256, 128}, 256, 128, 0);
  CheckIndexPutFirstDim({256, 128}, 256, 128, 128);
}
