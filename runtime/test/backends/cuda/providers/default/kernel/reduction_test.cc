//===- reduction_test.cc --------------------------------------*--- C++ -*-===//
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
#include "brt/core/common/utils/math_helper.h"
#include "brt/core/session/request_context.h"
#include "brt/core/session/session.h"
#include "brt/test/common/cuda/util.h"
#include "brt/test/common/models.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <cuda_runtime.h>
#include <memory>
#include <string>

using namespace brt;
using namespace brt::common;
using namespace brt::ir;
using namespace brt::test;

namespace {

std::vector<int64_t> decodeIndices(const std::vector<int64_t> &shape,
                                   size_t encoded) {
  std::vector<int64_t> indices(shape.size());
  auto j = indices.rbegin();
  for (auto i = shape.rbegin(); i != shape.rend(); ++i, ++j) {
    *j = encoded % (*i);
    encoded /= *i;
  }
  return indices;
}

template <typename T> struct ReduceSumOp {
  using type_t = T;
  ReduceSumOp() : acc_(0) {}
  void operator()(T val) { acc_ += val; }
  T get() { return acc_; }
  T acc_;
};

template <typename T> struct ReduceMaxOp {
  using type_t = T;
  ReduceMaxOp() : acc_(DTypeTraits<dtype_enum_v<T>>::lower_bound()) {}
  void operator()(T val) { acc_ = std::max(acc_, val); }
  T get() { return acc_; }
  T acc_;
};

template <typename ReduceOp, typename T = typename ReduceOp::type_t>
void GoldenReduction(T *h_input, T *h_output,
                     const std::vector<int64_t> &input_shape,
                     const std::vector<int64_t> &output_shape,
                     const std::vector<int64_t> &dimensions) {
  size_t input_size = LinearizedShape(input_shape);
  size_t output_size = LinearizedShape(output_shape);

  std::vector<ReduceOp> result(output_size);
  std::vector<bool> dim_mask(input_shape.size(), false);
  for (auto &&i : dimensions)
    dim_mask[i] = true;

  for (size_t inpIdx = 0; inpIdx < input_size; ++inpIdx) {
    auto indices = decodeIndices(input_shape, inpIdx);
    size_t oupIdx = 0;
    for (size_t i = 0; i < input_shape.size(); ++i) {
      if (!dim_mask[i])
        oupIdx = oupIdx * input_shape[i] + indices[i];
    }
    result[oupIdx](h_input[inpIdx]);
  }

  for (size_t i = 0; i < output_size; ++i) {
    h_output[i] = result[i].get();
  }
}

template <typename ReduceOp, typename T = typename ReduceOp::type_t>
void CheckReductionSingle(const std::vector<int64_t> &input_shape,
                          const std::vector<int64_t> &dimensions,
                          std::string op_name) {
  Session session;
  auto status_allocator = CUDAAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);
  auto status_cuda = DefaultCUDAExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cuda);

  ByREBuilder byre_builder;
  auto status_load = session.LoadFromMemory(
      CreateReduction(byre_builder, "cuda", input_shape, dimensions, op_name),
      "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  std::unique_ptr<RequestContext> request;
  auto status_request = session.NewRequestContext(&request);
  BRT_TEST_CHECK_STATUS(status_request);

  // prepare for checking values
  std::vector<int64_t> output_shape =
      brt::reduction::DeduceOutputShape(input_shape, dimensions);
  size_t input_size = LinearizedShape(input_shape);
  size_t input_size_in_bytes = input_size * sizeof(T);
  size_t output_size = LinearizedShape(output_shape);
  size_t output_size_in_bytes = output_size * sizeof(T);
  T *h_input = (T *)malloc(input_size_in_bytes);
  T *h_output = (T *)malloc(output_size_in_bytes);
  T *golden_output = (T *)malloc(output_size_in_bytes);

  // initiate input
  RandCUDABuffer((T *)request->GetArg(0), LinearizedShape(input_shape));

  request->FinishIOBinding();

  auto status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);

  auto status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);

  cudaMemcpy(h_input, request->GetArg(0), input_size_in_bytes,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_output, request->GetArg(1), output_size_in_bytes,
             cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  GoldenReduction<ReduceOp>(h_input, golden_output, input_shape, output_shape,
                            dimensions);
  bool passed =
      CheckCPUValues(golden_output, h_output, output_size, 1e-4f, 1e-5f);
  EXPECT_TRUE(passed);
  // second run
  RandCUDABuffer((T *)request->GetArg(0), LinearizedShape(input_shape));
  status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);

  status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);

  cudaMemcpy(h_input, request->GetArg(0), input_size_in_bytes,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_output, request->GetArg(1), output_size_in_bytes,
             cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  GoldenReduction<ReduceOp>(h_input, golden_output, input_shape, output_shape,
                            dimensions);
  passed = CheckCPUValues(golden_output, h_output, output_size, 1e-4f, 1e-5f);
  EXPECT_TRUE(passed);

  free(h_input);
  free(h_output);
  free(golden_output);
}

template <typename ReduceOp> void CheckReduction(std::string op_name) {
  CheckReductionSingle<ReduceOp>({1, 16, 32}, {0}, op_name);
  CheckReductionSingle<ReduceOp>({1, 16, 32}, {1}, op_name);
  CheckReductionSingle<ReduceOp>({1, 16, 32}, {2}, op_name);
  CheckReductionSingle<ReduceOp>({1, 16, 32}, {0, 1}, op_name);
  CheckReductionSingle<ReduceOp>({1, 16, 32}, {0, 2}, op_name);
  CheckReductionSingle<ReduceOp>({1, 128, 128}, {0, 1}, op_name);
  CheckReductionSingle<ReduceOp>({2, 128, 256}, {0, 1}, op_name);
  CheckReductionSingle<ReduceOp>({2, 16, 8, 64}, {1, 2}, op_name);
}

void CheckNanPropagation(std::string op_name) {
  const static size_t nr_elems = 1024;
  Session session;
  auto status_allocator = CUDAAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);
  auto status_cuda = DefaultCUDAExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cuda);

  ByREBuilder byre_builder;
  auto status_load = session.LoadFromMemory(
      CreateReduction(byre_builder, "cuda", {nr_elems}, {0}, op_name), "byre");

  std::unique_ptr<RequestContext> request;
  auto status_request = session.NewRequestContext(&request);
  BRT_TEST_CHECK_STATUS(status_request);

  float *device_ptr = static_cast<float *>(request->GetArg(0));
  RandCUDABuffer(device_ptr, nr_elems);
  // set the first element of input as NaN
  AssignCUDABuffer(device_ptr, 1, std::numeric_limits<float>::quiet_NaN());

  request->FinishIOBinding();

  auto status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);

  auto status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);

  float host_v = 0;
  cudaMemcpy(&host_v, request->GetArg(1), sizeof(float),
             cudaMemcpyDeviceToHost);
  ASSERT_TRUE(std::isnan(host_v));
}
} // namespace

TEST(CUDATestReductionOp, ReduceSum) {
  CheckReduction<ReduceSumOp<float>>("ReduceSumOp_f32_f32");
}

TEST(CUDATestReductionOp, ReduceMax) {
  CheckReduction<ReduceMaxOp<float>>("ReduceMaxOp_f32_f32");
  // TODO(liuyuanqiang): add fp16 test and resolve precision problem.
}

TEST(CUDATestReductionOp, NanPropagation) {
  CheckNanPropagation("ReduceSumOp_f32_f32");
  CheckNanPropagation("ReduceMaxOp_f32_f32");
}
