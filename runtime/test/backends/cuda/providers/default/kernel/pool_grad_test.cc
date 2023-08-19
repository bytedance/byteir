//===- pool_grad_test.cc --------------------------------------*--- C++ -*-===//
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
static void
GoldenPoolMaxGrad4_2D(T *x, T *dy, T *dx, const std::string &layout, int64_t N,
                      int64_t C, int64_t iH, int64_t iW, int64_t oH, int64_t oW,
                      int64_t winH, int64_t winW, int64_t strideH,
                      int64_t strideW, int64_t paddingH, int64_t paddingW) {
  T *h_dx_cpu = (T *)malloc(N * C * iH * iW * sizeof(T));
  // preset dx as 0
  memset(h_dx_cpu, 0, N * C * iH * iW * sizeof(T));

  for (int64_t n = 0; n < N; n++) {
    for (int64_t c = 0; c < C; c++) {
      for (int64_t oh = 0; oh < oH; oh++) {
        for (int64_t ow = 0; ow < oW; ow++) {
          int64_t y_index = 0;
          if (layout == "NHWC") {
            y_index = n * oH * oW * C + oh * oW * C + ow * C + c;
          } else if (layout == "NCHW") {
            y_index = n * C * oH * oW + c * oH * oW + oh * oW + ow;
          }
          T y = static_cast<T>(-std::numeric_limits<float>::infinity());
          int64_t ih = oh * strideH - paddingH;
          int64_t iw = ow * strideW - paddingW;
          int64_t x_final_index = -1;
          for (int64_t kh = 0; kh < winH; kh++) {
            for (int64_t kw = 0; kw < winW; kw++) {
              int64_t t_ih = ih + kh;
              int64_t t_iw = iw + kw;
              T t_x = static_cast<T>(-std::numeric_limits<float>::infinity());
              int64_t x_index = -1;
              if (t_ih >= 0 && t_iw >= 0 && t_ih < iH && t_iw < iW) {
                if (layout == "NHWC") {
                  x_index = n * iH * iW * C + t_ih * iW * C + t_iw * C + c;
                } else if (layout == "NCHW") {
                  x_index = n * C * iH * iW + c * iH * iW + t_ih * iW + t_iw;
                }
                t_x = x[x_index];
              }

              if (t_x > y) {
                y = t_x;
                x_final_index = x_index;
              }
            }
          }

          h_dx_cpu[x_final_index] = h_dx_cpu[x_final_index] + dy[y_index];
        }
      }

      for (int64_t ih = 0; ih < iH; ih++) {
        for (int64_t iw = 0; iw < iW; iw++) {
          int64_t x_index = -1;
          if (layout == "NHWC") {
            x_index = n * iH * iW * C + ih * iW * C + iw * C + c;
          } else if (layout == "NCHW") {
            x_index = n * C * iH * iW + c * iH * iW + ih * iW + iw;
          }
          dx[x_index] = h_dx_cpu[x_index];
        }
      }
    }
  }

  free(h_dx_cpu);
}

template <typename T>
static void
TestPoolMaxGradOp2D(std::vector<int64_t> shape_x, std::vector<int64_t> shape_y,
                    std::vector<int64_t> padding,
                    std::vector<int64_t> window_dimensions,
                    std::vector<int64_t> window_strides,
                    const std::string &layout, float abs_eps, float rel_eps) {
  ByREBuilder byre_builder;
  Session session;
  auto status_allocator = CUDAAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);
  auto status_cuda = DefaultCUDAExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cuda);

  auto status_load = session.LoadFromMemory(
      CreatePoolMaxGrad(byre_builder, dtype_enum_v<T>, "cuda", shape_x, shape_y,
                        padding, window_dimensions, window_strides),
      "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  std::unique_ptr<RequestContext> request;
  auto status_request = session.NewRequestContext(&request);
  BRT_TEST_CHECK_STATUS(status_request);

  auto x_total_size = LinearizedShape(shape_x);
  auto y_total_size = LinearizedShape(shape_y);
  BRT_ENFORCE(shape_y == pool::DeduceOutputShape(shape_x, window_dimensions,
                                                 window_strides, padding));

  int64_t N, C, iH, iW, oH, oW, kH, kW, strideH, strideW, paddingH, paddingW;
  if (layout == "NHWC") {
    N = shape_x[0];
    EXPECT_EQ(N, shape_y[0]);
    C = shape_x[3];
    EXPECT_EQ(C, shape_y[3]);
    iH = shape_x[1];
    iW = shape_x[2];
    oH = shape_y[1];
    oW = shape_y[2];
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
    N = shape_x[0];
    EXPECT_EQ(N, shape_y[0]);
    C = shape_x[1];
    EXPECT_EQ(C, shape_y[1]);
    iH = shape_x[2];
    iW = shape_x[3];
    oH = shape_y[2];
    oW = shape_y[3];
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

  // initiate x
  T *d_x = (T *)request->GetArg(0);
  RandCUDABuffer(d_x, x_total_size, -10.f, 10.f);

  T *d_dy = (T *)request->GetArg(1);
  RandCUDABuffer(d_dy, y_total_size, -10.f, 10.f);

  // I/O binding
  request->FinishIOBinding();

  auto status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);

  auto status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);

  T *d_dx = (T *)request->GetArg(2);

  // check result
  T *h_x = (T *)malloc(x_total_size * sizeof(T));
  T *h_dy = (T *)malloc(y_total_size * sizeof(T));
  T *h_dx = (T *)malloc(x_total_size * sizeof(T));
  cudaMemcpy(h_x, d_x, x_total_size * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_dy, d_dy, y_total_size * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_dx, d_dx, x_total_size * sizeof(T), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  T *golden_dx = (T *)malloc(x_total_size * sizeof(T));

  GoldenPoolMaxGrad4_2D<T>(h_x, h_dy, golden_dx, layout, N, C, iH, iW, oH, oW,
                           kH, kW, strideH, strideW, paddingH, paddingW);
  bool passed = CheckCPUValues(h_dx, golden_dx, x_total_size, abs_eps, rel_eps);
  EXPECT_TRUE(passed);

  free(h_x);
  free(h_dy);
  free(h_dx);
  free(golden_dx);
}

TEST(CUDAOpKernelTest, PoolMaxGradOp2D) {
  float abs_eps = 1e-6f, rel_eps = 1e-5f;
  // small case for debug purpose
  TestPoolMaxGradOp2D<float>(/*shape_input=*/{1, 1, 3, 3},
                             /*shape_output=*/{1, 1, 2, 2},
                             /*padding=*/{0, 0, 0, 0, 1, 1, 1, 1},
                             /*window_dimensions*/ {1, 1, 3, 3},
                             /*window_strides*/ {1, 1, 2, 2}, "NCHW", abs_eps,
                             rel_eps);
  TestPoolMaxGradOp2D<float>(/*shape_input=*/{1, 64, 112, 112},
                             /*shape_output=*/{1, 64, 56, 56},
                             /*padding=*/{0, 0, 0, 0, 1, 1, 1, 1},
                             /*window_dimensions*/ {1, 1, 3, 3},
                             /*window_strides*/ {1, 1, 2, 2}, "NCHW", abs_eps,
                             rel_eps);
  TestPoolMaxGradOp2D<float>(/*shape_input=*/{32, 64, 3, 3},
                             /*shape_output=*/{32, 64, 1, 1},
                             /*padding=*/{0, 0, 0, 0, 0, 0, 0, 0},
                             /*window_dimensions*/ {1, 1, 3, 3},
                             /*window_strides*/ {1, 1, 2, 2}, "NCHW", abs_eps,
                             rel_eps);
  TestPoolMaxGradOp2D<float>(/*shape_input=*/{32, 64, 3, 3},
                             /*shape_output=*/{32, 64, 1, 3},
                             /*padding=*/{0, 0, 0, 0, 0, 0, 0, 0},
                             /*window_dimensions*/ {1, 1, 3, 1},
                             /*window_strides*/ {1, 1, 2, 1}, "NCHW", abs_eps,
                             rel_eps);
  TestPoolMaxGradOp2D<float>(/*shape_input=*/{1, 1, 1, 3},
                             /*shape_output=*/{1, 1, 1, 1},
                             /*padding=*/{0, 0, 0, 0, 0, 0, 0, 0},
                             /*window_dimensions*/ {1, 1, 1, 3},
                             /*window_strides*/ {1, 1, 1, 2}, "NCHW", abs_eps,
                             rel_eps);
  TestPoolMaxGradOp2D<float>(/*shape_input=*/{1, 112, 112, 64},
                             /*shape_output=*/{1, 56, 56, 64},
                             /*padding=*/{0, 0, 1, 1, 1, 1, 0, 0},
                             /*window_dimensions*/ {1, 3, 3, 1},
                             /*window_strides*/ {1, 2, 2, 1}, "NHWC", abs_eps,
                             rel_eps);
  TestPoolMaxGradOp2D<float>(/*shape_input=*/{32, 3, 3, 64},
                             /*shape_output=*/{32, 1, 1, 64},
                             /*padding=*/{0, 0, 0, 0, 0, 0, 0, 0},
                             /*window_dimensions*/ {1, 3, 3, 1},
                             /*window_strides*/ {1, 2, 2, 1}, "NHWC", abs_eps,
                             rel_eps);
}

TEST(CUDAOpKernelTest, PoolMaxGradOp2DFp16) {
  float abs_eps = 3e-2f, rel_eps = 3e-3f;
  TestPoolMaxGradOp2D<__half>(/*shape_input=*/{1, 64, 112, 112},
                              /*shape_output=*/{1, 64, 56, 56},
                              /*padding=*/{0, 0, 0, 0, 1, 1, 1, 1},
                              /*window_dimensions*/ {1, 1, 3, 3},
                              /*window_strides*/ {1, 1, 2, 2}, "NCHW", abs_eps,
                              rel_eps);
  TestPoolMaxGradOp2D<__half>(/*shape_input=*/{32, 64, 3, 3},
                              /*shape_output=*/{32, 64, 1, 1},
                              /*padding=*/{0, 0, 0, 0, 0, 0, 0, 0},
                              /*window_dimensions*/ {1, 1, 3, 3},
                              /*window_strides*/ {1, 1, 2, 2}, "NCHW", abs_eps,
                              rel_eps);
  TestPoolMaxGradOp2D<__half>(/*shape_input=*/{32, 64, 3, 3},
                              /*shape_output=*/{32, 64, 1, 3},
                              /*padding=*/{0, 0, 0, 0, 0, 0, 0, 0},
                              /*window_dimensions*/ {1, 1, 3, 1},
                              /*window_strides*/ {1, 1, 2, 1}, "NCHW", abs_eps,
                              rel_eps);
  TestPoolMaxGradOp2D<__half>(/*shape_input=*/{1, 1, 1, 3},
                              /*shape_output=*/{1, 1, 1, 1},
                              /*padding=*/{0, 0, 0, 0, 0, 0, 0, 0},
                              /*window_dimensions*/ {1, 1, 1, 3},
                              /*window_strides*/ {1, 1, 1, 2}, "NCHW", abs_eps,
                              rel_eps);
  TestPoolMaxGradOp2D<__half>(/*shape_input=*/{1, 112, 112, 64},
                              /*shape_output=*/{1, 56, 56, 64},
                              /*padding=*/{0, 0, 1, 1, 1, 1, 0, 0},
                              /*window_dimensions*/ {1, 3, 3, 1},
                              /*window_strides*/ {1, 2, 2, 1}, "NHWC", abs_eps,
                              rel_eps);
  TestPoolMaxGradOp2D<__half>(/*shape_input=*/{32, 3, 3, 64},
                              /*shape_output=*/{32, 1, 1, 64},
                              /*padding=*/{0, 0, 0, 0, 0, 0, 0, 0},
                              /*window_dimensions*/ {1, 3, 3, 1},
                              /*window_strides*/ {1, 2, 2, 1}, "NHWC", abs_eps,
                              rel_eps);
}
