//===- conv_test.cc -------------------------------------------*--- C++ -*-===//
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
#include "brt/core/framework/dtype.h"
#include "brt/core/session/request_context.h"
#include "brt/core/session/session.h"
#include "brt/test/common/cuda/util.h"
#include "brt/test/common/models.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>

using namespace std;
using namespace brt;
using namespace brt::common;
using namespace brt::ir;
using namespace brt::test;

template <typename T, typename CompType = float>
static void GoldenConv(T *input, T *filter, T *output, const string &layout,
                       int64_t N, int64_t iC, int64_t iH, int64_t iW,
                       int64_t oC, int64_t kH, int64_t kW, int64_t oH,
                       int64_t oW, int64_t strideH, int64_t strideW,
                       int64_t paddingH, int64_t paddingW, int64_t dilateH,
                       int64_t dilateW) {
  EXPECT_EQ(dilateH, 1);
  EXPECT_EQ(dilateW, 1);
  // if (dilateH != 1 || dilateW != 1) {
  //   int64_t _kH = kH * dilateH - dilateH + 1;
  //   int64_t _kW = kW * dilateW - dilateW + 1;
  //   T *filter_temp = (T *)malloc(oC * _kH * _kW * iC * sizeof(T));
  //   memset(filter_temp, 0, oC * _kH * _kW * iC * sizeof(T));
  //   for (int64_t oc = 0; oc < oC; oc++) {
  //     for (int64_t ic = 0; ic < iC; ic++) {
  //       for (int64_t kh = 0; kh < kH; kh++) {
  //         for (int64_t kw = 0; kw < kW; kw++) {
  //           if (layout == "NHWC") {
  //             filter_temp[oc * _kH * _kW * iC + (kh * dilateH + 1) * _kW * iC
  //             +
  //                         (kw * dilateW + 1) * iC + ic] =
  //                 filter[oc * kH * kW * iC + kh * kW * iC + kw * iC + ic];
  //           } else if (layout == "NCHW") {
  //             filter_temp[oc * iC * _kH * _kW + ic * _kH * _kW +
  //                         (kh * dilateH + 1) * _kW + (kw * dilateW + 1)] =
  //                 filter[oc * iC * kH * kW + ic * kH * kW + kh * kW + kw];
  //           }
  //         }
  //       }
  //     }
  //   }
  //   free(filter);
  //   filter = filter_temp;
  //   kH = _kH;
  //   kW = _kW;
  // }

  for (int64_t n = 0; n < N; n++) {
    for (int64_t oc = 0; oc < oC; oc++) {
      for (int64_t oh = 0; oh < oH; oh++) {
        for (int64_t ow = 0; ow < oW; ow++) {
          int64_t output_index = 0;
          if (layout == "NHWC") {
            output_index = n * oH * oW * oC + oh * oW * oC + ow * oC + oc;
          } else if (layout == "NCHW") {
            output_index = n * oC * oH * oW + oc * oH * oW + oh * oW + ow;
          }
          CompType result = static_cast<CompType>(0.f);
          for (int64_t ic = 0; ic < iC; ic++) {
            int64_t ih = oh * strideH - paddingH;
            int64_t iw = ow * strideW - paddingW;
            for (int64_t kh = 0; kh < kH; kh++) {
              for (int64_t kw = 0; kw < kW; kw++) {
                int64_t filter_index = 0;
                if (layout == "NHWC") {
                  filter_index =
                      oc * kH * kW * iC + kh * kW * iC + kw * iC + ic;
                } else if (layout == "NCHW") {
                  filter_index =
                      oc * iC * kH * kW + ic * kH * kW + kh * kW + kw;
                }
                int64_t t_ih = ih + kh;
                int64_t t_iw = iw + kw;
                if (t_ih >= 0 && t_iw >= 0 && t_ih < iH && t_iw < iW) {
                  int64_t input_index = 0;
                  if (layout == "NHWC") {
                    input_index =
                        n * iH * iW * iC + t_ih * iW * iC + t_iw * iC + ic;
                  } else if (layout == "NCHW") {
                    input_index =
                        n * iC * iH * iW + ic * iH * iW + t_ih * iW + t_iw;
                  }
                  result += static_cast<CompType>(input[input_index]) *
                            static_cast<CompType>(filter[filter_index]);
                }
              }
            }
          }
          output[output_index] = static_cast<T>(result);
        }
      }
    }
  }
}

template <typename T>
static void TestConvOp(float abs_eps, float rel_eps, const std::string &layout,
                       int64_t N, int64_t iC, int64_t iH, int64_t iW,
                       int64_t oC, int64_t kH, int64_t kW, int64_t strideH,
                       int64_t strideW, int64_t paddingH, int64_t paddingW,
                       int64_t dilateH = 1, int64_t dilateW = 1) {
  auto dtype = dtype_enum_v<T>;

  Session session;
  auto status_allocator = CUDAAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);
  auto status_cuda = DefaultCUDAExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cuda);

  ByREBuilder byre_builder;
  auto status_load = session.LoadFromMemory(
      CreateConv(byre_builder, "ConvOp", dtype, "cuda", N, iC, iH, iW, oC, kH,
                 kW, layout, strideH, strideW, paddingH, paddingW, dilateH,
                 dilateW),
      "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  std::unique_ptr<RequestContext> request;
  auto status_request = session.NewRequestContext(&request);
  BRT_TEST_CHECK_STATUS(status_request);

  auto shape_input = session.GetStaticShape(0);
  auto input_total_size = LinearizedShape(shape_input);
  auto shape_filter = session.GetStaticShape(1);
  auto filter_total_size = LinearizedShape(shape_filter);
  auto shape_output = session.GetStaticShape(2);
  auto output_total_size = LinearizedShape(shape_output);
  int64_t oH, oW;
  if (layout == "NHWC") {
    EXPECT_EQ(N, shape_input[0]);
    EXPECT_EQ(iC, shape_input[3]);
    EXPECT_EQ(iH, shape_input[1]);
    EXPECT_EQ(iW, shape_input[2]);
    EXPECT_EQ(oC, shape_output[3]);
    oH = shape_output[1];
    oW = shape_output[2];
    EXPECT_EQ(kH, shape_filter[1]);
    EXPECT_EQ(kW, shape_filter[2]);
  } else if (layout == "NCHW") {
    EXPECT_EQ(N, shape_input[0]);
    EXPECT_EQ(iC, shape_input[1]);
    EXPECT_EQ(iH, shape_input[2]);
    EXPECT_EQ(iW, shape_input[3]);
    EXPECT_EQ(oC, shape_output[1]);
    oH = shape_output[2];
    oW = shape_output[3];
    EXPECT_EQ(kH, shape_filter[2]);
    EXPECT_EQ(kW, shape_filter[3]);
  } else {
    BRT_THROW("invalid conv layout");
  }

  // initiate A
  T *d_A = (T *)request->GetArg(0);
  RandCUDABuffer(d_A, input_total_size, 0.f, 1.f);
  // initiate B
  T *d_B = (T *)request->GetArg(1);
  RandCUDABuffer(d_B, filter_total_size, 0.f, 1.f);

  // I/O binding
  request->FinishIOBinding();

  auto status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);

  auto status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);

  T *d_C = (T *)request->GetArg(2);

  // check result
  T *h_A = (T *)malloc(input_total_size * sizeof(T));
  T *h_B = (T *)malloc(filter_total_size * sizeof(T));
  T *h_C = (T *)malloc(output_total_size * sizeof(T));
  cudaMemcpy(h_A, d_A, input_total_size * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_B, d_B, filter_total_size * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_C, d_C, output_total_size * sizeof(T), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  T *golden_C = (T *)malloc(output_total_size * sizeof(T));
  GoldenConv<T>(h_A, h_B, golden_C, layout, N, iC, iH, iW, oC, kH, kW, oH, oW,
                strideH, strideW, paddingH, paddingW, dilateH, dilateW);

  bool passed =
      CheckCPUValues<T>(golden_C, h_C, output_total_size, abs_eps, rel_eps);
  EXPECT_TRUE(passed) << "layout: " << layout << " N: " << N << " iC: " << iC
                      << " iH: " << iH << " iW: " << iW << " oC: " << oC
                      << " kH: " << kH << " kW: " << kW
                      << " strideH: " << strideH << " strideW: " << strideW
                      << " paddingH: " << paddingH << "paddingW: " << paddingW
                      << "\n";

  free(h_A);
  free(h_B);
  free(h_C);
  free(golden_C);
}

TEST(CUDAOpKernelTest, ConvOp) {
  float abs_eps = 1e-4f, rel_eps = 1e-4f;
  TestConvOp<float>(abs_eps, rel_eps, /*layout=*/"NHWC", /*N=*/1, /*iC=*/64,
                    /*iH=*/28,
                    /*iW=*/28, /*oC=*/128, /*kH=*/3, /*kW=*/3, /*strideH=*/2,
                    /*strideW=*/2, /*paddingH=*/1, /*paddingW=*/1);
  TestConvOp<float>(abs_eps, rel_eps, /*layout=*/"NCHW", /*N=*/1, /*iC=*/64,
                    /*iH=*/28,
                    /*iW=*/28, /*oC=*/128, /*kH=*/3, /*kW=*/3, /*strideH=*/2,
                    /*strideW=*/2, /*paddingH=*/1, /*paddingW=*/1);
  TestConvOp<float>(abs_eps, rel_eps, /*layout=*/"NCHW", /*N=*/1, /*iC=*/64,
                    /*iH=*/28,
                    /*iW=*/28, /*oC=*/128, /*kH=*/3, /*kW=*/3, /*strideH=*/1,
                    /*strideW=*/1, /*paddingH=*/0, /*paddingW=*/0);
  TestConvOp<float>(abs_eps, rel_eps, /*layout=*/"NCHW", /*N=*/1, /*iC=*/64,
                    /*iH=*/28,
                    /*iW=*/28, /*oC=*/128, /*kH=*/3, /*kW=*/5, /*strideH=*/2,
                    /*strideW=*/2, /*paddingH=*/1, /*paddingW=*/0);
  // TestConvOp<float>(/*layout=*/"NHWC", /*N=*/1, /*iC=*/64, /*iH=*/28,
  //                   /*iW=*/28, /*oC=*/128, /*kH=*/3, /*kW=*/3, /*strideH=*/2,
  //                   /*strideW=*/2, /*paddingH=*/1, /*paddingW=*/1,
  //                   /*dilateH=*/2, /*dilateW=*/2);
}

TEST(CUDAOpKernelTest, ConvOpFp16NHWC) {
  float abs_eps = 1e-2f, rel_eps = 1e-2f;
  TestConvOp<__half>(abs_eps, rel_eps, /*layout=*/"NHWC", /*N=*/1, /*iC=*/3,
                     /*iH=*/224,
                     /*iW=*/224, /*oC=*/64, /*kH=*/7, /*kW=*/7,
                     /*strideH=*/2, /*strideW=*/2,
                     /*paddingH=*/3, /*paddingW=*/3);
  TestConvOp<__half>(abs_eps, rel_eps, /*layout=*/"NHWC", /*N=*/1, /*iC=*/64,
                     /*iH=*/28,
                     /*iW=*/28, /*oC=*/128, /*kH=*/3, /*kW=*/3,
                     /*strideH=*/2, /*strideW=*/2,
                     /*paddingH=*/1, /*paddingW=*/1);
  TestConvOp<__half>(abs_eps, rel_eps, /*layout=*/"NHWC", /*N=*/1, /*iC=*/64,
                     /*iH=*/28,
                     /*iW=*/28, /*oC=*/128, /*kH=*/3, /*kW=*/3,
                     /*strideH=*/1, /*strideW=*/1,
                     /*paddingH=*/0, /*paddingW=*/0);
  TestConvOp<__half>(abs_eps, rel_eps, /*layout=*/"NHWC", /*N=*/1, /*iC=*/64,
                     /*iH=*/28,
                     /*iW=*/28, /*oC=*/128, /*kH=*/3, /*kW=*/5,
                     /*strideH=*/2, /*strideW=*/2,
                     /*paddingH=*/3, /*paddingW=*/0);
}

TEST(CUDAOpKernelTest, ConvOpFp16NCHW) {
  float abs_eps = 1e-2f, rel_eps = 1e-2f;
  TestConvOp<__half>(abs_eps, rel_eps, /*layout=*/"NCHW", /*N=*/1, /*iC=*/3,
                     /*iH=*/224,
                     /*iW=*/224, /*oC=*/64, /*kH=*/7, /*kW=*/7,
                     /*strideH=*/2, /*strideW=*/2,
                     /*paddingH=*/3, /*paddingW=*/3);
  TestConvOp<__half>(abs_eps, rel_eps, /*layout=*/"NCHW", /*N=*/1, /*iC=*/64,
                     /*iH=*/28,
                     /*iW=*/28, /*oC=*/128, /*kH=*/3, /*kW=*/3,
                     /*strideH=*/2, /*strideW=*/2,
                     /*paddingH=*/1, /*paddingW=*/1);
}
