//===- pool_test.cc -------------------------------------------*--- C++ -*-===//
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

#include "brt/backends/cuda/device/common/dtype.h"
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
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace std;
using namespace brt;
using namespace brt::common;
using namespace brt::ir;
using namespace brt::test;

template <typename T>
static void GoldenPoolMax4_2D(T *input, T *output, const std::string &layout,
                              int64_t N, int64_t C, int64_t iH, int64_t iW,
                              int64_t oH, int64_t oW, int64_t kH, int64_t kW,
                              int64_t strideH, int64_t strideW,
                              int64_t paddingH, int64_t paddingW) {
  for (int64_t n = 0; n < N; n++) {
    for (int64_t c = 0; c < C; c++) {
      for (int64_t oh = 0; oh < oH; oh++) {
        for (int64_t ow = 0; ow < oW; ow++) {
          int64_t output_index = 0;
          if (layout == "NHWC") {
            output_index = n * oH * oW * C + oh * oW * C + ow * C + c;
          } else if (layout == "NCHW") {
            output_index = n * C * oH * oW + c * oH * oW + oh * oW + ow;
          }
          T result = static_cast<T>(-std::numeric_limits<float>::infinity());
          int64_t ih = oh * strideH - paddingH;
          int64_t iw = ow * strideW - paddingW;
          for (int64_t kh = 0; kh < kH; kh++) {
            for (int64_t kw = 0; kw < kW; kw++) {
              int64_t t_ih = ih + kh;
              int64_t t_iw = iw + kw;
              T t_input =
                  static_cast<T>(-std::numeric_limits<float>::infinity());
              if (t_ih >= 0 && t_iw >= 0 && t_ih < iH && t_iw < iW) {
                int64_t input_index = 0;
                if (layout == "NHWC") {
                  input_index = n * iH * iW * C + t_ih * iW * C + t_iw * C + c;
                } else if (layout == "NCHW") {
                  input_index =
                      n * C * iH * iW + c * iH * iW + t_ih * iW + t_iw;
                }
                t_input = input[input_index];
              }
              result = max(result, t_input);
            }
          }
          output[output_index] = result;
        }
      }
    }
  }
}

template <typename T>
static void TestPoolMaxOp2D(std::vector<int64_t> shape_input,
                            std::vector<int64_t> shape_output,
                            std::vector<int64_t> padding,
                            std::vector<int64_t> window_dimensions,
                            std::vector<int64_t> window_strides,
                            const std::string &layout) {
  ByREBuilder byre_builder;
  Session session;
  auto status_allocator = CUDAAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);
  auto status_cuda = DefaultCUDAExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cuda);

  auto status_load = session.LoadFromMemory(
      CreatePoolMax(byre_builder, dtype_enum_v<T>, "cuda", shape_input,
                    shape_output, padding, window_dimensions, window_strides),
      "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  std::unique_ptr<RequestContext> request;
  auto status_request = session.NewRequestContext(&request);
  BRT_TEST_CHECK_STATUS(status_request);

  auto shape_A = session.GetStaticShape(0);
  auto input_total_size = LinearizedShape(shape_input);
  auto shape_B = session.GetStaticShape(1);
  auto output_total_size = LinearizedShape(shape_output);
  BRT_ENFORCE(shape_A == shape_input && shape_B == shape_output);
  BRT_ENFORCE(shape_output == pool::DeduceOutputShape(shape_input,
                                                      window_dimensions,
                                                      window_strides, padding));

  int64_t N, C, iH, iW, oH, oW, kH, kW, strideH, strideW, paddingH, paddingW;
  if (layout == "NHWC") {
    N = shape_A[0];
    EXPECT_EQ(N, shape_B[0]);
    C = shape_A[3];
    EXPECT_EQ(C, shape_B[3]);
    iH = shape_A[1];
    iW = shape_A[2];
    oH = shape_B[1];
    oW = shape_B[2];
    kH = window_dimensions[1];
    kW = window_dimensions[2];
    EXPECT_EQ(window_dimensions[0], 1);
    EXPECT_EQ(window_dimensions[3], 1);
    strideH = window_strides[1];
    strideW = window_strides[2];
    EXPECT_EQ(window_strides[0], 1);
    EXPECT_EQ(window_strides[3], 1);
    paddingH = padding[2];
    EXPECT_EQ(padding[2], padding[3]);
    paddingW = padding[4];
    EXPECT_EQ(padding[4], padding[5]);
    EXPECT_EQ(padding[0], 0);
    EXPECT_EQ(padding[1], 0);
    EXPECT_EQ(padding[6], 0);
    EXPECT_EQ(padding[7], 0);
  } else if (layout == "NCHW") {
    N = shape_A[0];
    EXPECT_EQ(N, shape_B[0]);
    C = shape_A[1];
    EXPECT_EQ(C, shape_B[1]);
    iH = shape_A[2];
    iW = shape_A[3];
    oH = shape_B[2];
    oW = shape_B[3];
    kH = window_dimensions[2];
    kW = window_dimensions[3];
    EXPECT_EQ(window_dimensions[0], 1);
    EXPECT_EQ(window_dimensions[1], 1);
    strideH = window_strides[2];
    strideW = window_strides[3];
    EXPECT_EQ(window_strides[0], 1);
    EXPECT_EQ(window_strides[1], 1);
    paddingH = padding[4];
    EXPECT_EQ(padding[4], padding[5]);
    paddingW = padding[6];
    EXPECT_EQ(padding[6], padding[7]);
    EXPECT_EQ(padding[0], 0);
    EXPECT_EQ(padding[1], 0);
    EXPECT_EQ(padding[2], 0);
    EXPECT_EQ(padding[3], 0);
  } else {
    BRT_THROW("unsupported pool layout");
  }

  // initiate A
  T *d_A = (T *)request->GetArg(0);
  RandCUDABuffer(d_A, input_total_size, -10.f, 10.f);

  // I/O binding
  request->FinishIOBinding();

  auto status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);

  auto status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);

  T *d_B = (T *)request->GetArg(1);

  // check result
  T *h_A = (T *)malloc(input_total_size * sizeof(T));
  T *h_B = (T *)malloc(output_total_size * sizeof(T));
  cudaMemcpy(h_A, d_A, input_total_size * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_B, d_B, output_total_size * sizeof(T), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  T *golden_B = (T *)malloc(output_total_size * sizeof(T));

  GoldenPoolMax4_2D<T>(h_A, golden_B, layout, N, C, iH, iW, oH, oW, kH, kW,
                       strideH, strideW, paddingH, paddingW);
  bool passed = CheckCPUValues(h_B, golden_B, output_total_size);
  EXPECT_TRUE(passed);

  free(h_A);
  free(h_B);
  free(golden_B);
}

TEST(CUDAOpKernelTest, PoolMaxOp2D) {
  // TODO: use pool::DeduceOutputShape as shape_output
  TestPoolMaxOp2D<float>(/*shape_input=*/{1, 64, 112, 112},
                         /*shape_output=*/{1, 64, 56, 56},
                         /*padding=*/{0, 0, 0, 0, 1, 1, 1, 1},
                         /*window_dimensions*/ {1, 1, 3, 3},
                         /*window_strides*/ {1, 1, 2, 2}, "NCHW");
  TestPoolMaxOp2D<float>(/*shape_input=*/{32, 64, 3, 3},
                         /*shape_output=*/{32, 64, 1, 1},
                         /*padding=*/{0, 0, 0, 0, 0, 0, 0, 0},
                         /*window_dimensions*/ {1, 1, 3, 3},
                         /*window_strides*/ {1, 1, 2, 2}, "NCHW");
  TestPoolMaxOp2D<float>(/*shape_input=*/{32, 64, 3, 3},
                         /*shape_output=*/{32, 64, 1, 3},
                         /*padding=*/{0, 0, 0, 0, 0, 0, 0, 0},
                         /*window_dimensions*/ {1, 1, 3, 1},
                         /*window_strides*/ {1, 1, 2, 1}, "NCHW");
  TestPoolMaxOp2D<float>(/*shape_input=*/{1, 1, 1, 3},
                         /*shape_output=*/{1, 1, 1, 1},
                         /*padding=*/{0, 0, 0, 0, 0, 0, 0, 0},
                         /*window_dimensions*/ {1, 1, 1, 3},
                         /*window_strides*/ {1, 1, 1, 2}, "NCHW");
  TestPoolMaxOp2D<float>(/*shape_input=*/{1, 112, 112, 64},
                         /*shape_output=*/{1, 56, 56, 64},
                         /*padding=*/{0, 0, 1, 1, 1, 1, 0, 0},
                         /*window_dimensions*/ {1, 3, 3, 1},
                         /*window_strides*/ {1, 2, 2, 1}, "NHWC");
  TestPoolMaxOp2D<float>(/*shape_input=*/{32, 3, 3, 64},
                         /*shape_output=*/{32, 1, 1, 64},
                         /*padding=*/{0, 0, 0, 0, 0, 0, 0, 0},
                         /*window_dimensions*/ {1, 3, 3, 1},
                         /*window_strides*/ {1, 2, 2, 1}, "NHWC");
}

TEST(CUDAOpKernelTest, PoolMaxOp2DFp16) {
  TestPoolMaxOp2D<__half>(/*shape_input=*/{1, 64, 112, 112},
                          /*shape_output=*/{1, 64, 56, 56},
                          /*padding=*/{0, 0, 0, 0, 1, 1, 1, 1},
                          /*window_dimensions*/ {1, 1, 3, 3},
                          /*window_strides*/ {1, 1, 2, 2}, "NCHW");
  TestPoolMaxOp2D<__half>(/*shape_input=*/{32, 64, 3, 3},
                          /*shape_output=*/{32, 64, 1, 1},
                          /*padding=*/{0, 0, 0, 0, 0, 0, 0, 0},
                          /*window_dimensions*/ {1, 1, 3, 3},
                          /*window_strides*/ {1, 1, 2, 2}, "NCHW");
  TestPoolMaxOp2D<__half>(/*shape_input=*/{32, 64, 3, 3},
                          /*shape_output=*/{32, 64, 1, 3},
                          /*padding=*/{0, 0, 0, 0, 0, 0, 0, 0},
                          /*window_dimensions*/ {1, 1, 3, 1},
                          /*window_strides*/ {1, 1, 2, 1}, "NCHW");
  TestPoolMaxOp2D<__half>(/*shape_input=*/{1, 1, 1, 3},
                          /*shape_output=*/{1, 1, 1, 1},
                          /*padding=*/{0, 0, 0, 0, 0, 0, 0, 0},
                          /*window_dimensions*/ {1, 1, 1, 3},
                          /*window_strides*/ {1, 1, 1, 2}, "NCHW");
  TestPoolMaxOp2D<__half>(/*shape_input=*/{1, 112, 112, 64},
                          /*shape_output=*/{1, 56, 56, 64},
                          /*padding=*/{0, 0, 1, 1, 1, 1, 0, 0},
                          /*window_dimensions*/ {1, 3, 3, 1},
                          /*window_strides*/ {1, 2, 2, 1}, "NHWC");
  TestPoolMaxOp2D<__half>(/*shape_input=*/{32, 3, 3, 64},
                          /*shape_output=*/{32, 1, 1, 64},
                          /*padding=*/{0, 0, 0, 0, 0, 0, 0, 0},
                          /*window_dimensions*/ {1, 3, 3, 1},
                          /*window_strides*/ {1, 2, 2, 1}, "NHWC");
}
