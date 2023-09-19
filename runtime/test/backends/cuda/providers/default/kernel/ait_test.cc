//===- ait_test.cc --------------------------------------------*--- C++ -*-===//
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

#include "brt/backends/cuda/device/common/cuda_call.h"
#include "brt/backends/cuda/device/cuda_allocator.h"
#include "brt/backends/cuda/device/cuda_work_queue.h"
#include "brt/backends/cuda/providers/default/cuda_provider.h"
#include "brt/core/common/status.h"
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
using namespace brt::cuda;
using namespace brt::ir;
using namespace brt::test;
using namespace mlir;
using namespace std;

static std::string test_file_ait =
    "test/test_files/AITOp/bmm_permute_entry.mlir";

static void CheckBMMPermute(float *d_A, float *d_B, float *d_C,
                            size_t batch_count, size_t m, size_t n, size_t k,
                            size_t d1, float eps) {
  float *h_A = (float *)malloc(batch_count * m * k * sizeof(float));
  float *h_B = (float *)malloc(batch_count * k * n * sizeof(float));
  float *h_C = (float *)malloc(batch_count * m * n * sizeof(float));
  cudaMemcpy(h_A, d_A, batch_count * m * k * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_B, d_B, batch_count * k * n * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_C, d_C, batch_count * m * n * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  size_t d0 = batch_count / d1;
  for (size_t b = 0; b < batch_count; b++) {
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
        float sum = 0.0f;
        for (size_t l = 0; l < k; l++) {
          sum += h_A[b * m * k + i * k + l] * h_B[b * k * n + l * n + j];
        }
        size_t b0 = b / d1, b1 = b % d1;
        EXPECT_NEAR(h_C[b0 * m * d1 * n + i * d1 * n + b1 * n + j], sum, eps);
      }
    }
  }

  free(h_A);
  free(h_B);
  free(h_C);
}

TEST(CUDATestAITOp, AITOp_BMM_Permute) {
  // this test can execute on A100 GPU only
  /*
  int nr_device;
  BRT_CUDA_CHECK(cudaGetDeviceCount(&nr_device));
  if (!nr_device)
    return;

  for (int i = 0; i < nr_device; ++i) {
    BRT_CUDA_CHECK(cudaSetDevice(i));

    Session session;
    auto status_allocator = CUDAAllocatorFactory(&session, i);
    BRT_TEST_CHECK_STATUS(status_allocator);
    auto status_cuda = DefaultCUDAExecutionProviderFactory(&session, i);
    BRT_TEST_CHECK_STATUS(status_cuda);

    ByREBuilder byre_builder;
    auto status_load = session.Load(test_file_ait, "byre");
    BRT_TEST_CHECK_STATUS(status_load);

    std::unique_ptr<RequestContext> request;
    auto status_request = session.NewRequestContext(&request);
    BRT_TEST_CHECK_STATUS(status_request);
    request->SetWorkQueue(new CUDAWorkQueue(i));

    auto shape_A = session.GetStaticShape(0);
    auto shape_B = session.GetStaticShape(1);
    auto shape_C = session.GetStaticShape(2);
    EXPECT_EQ(shape_A.size(), 3);
    EXPECT_EQ(shape_B.size(), 3);
    EXPECT_EQ(shape_C.size(), 4);

    size_t b = shape_A[0];
    size_t m = shape_A[1];
    size_t k = shape_A[2];
    size_t n = shape_B[2];
    EXPECT_EQ(b, shape_B[0]);
    EXPECT_EQ(k, shape_B[1]);
    EXPECT_EQ(k, shape_B[1]);
    EXPECT_EQ(m, shape_C[1]);
    EXPECT_EQ(b, shape_C[0] * shape_C[2]);
    EXPECT_EQ(n, shape_C[3]);
    size_t d1 = shape_C[1];

    // initiate A
    float *d_A = (float *)request->GetArg(0);
    RandCUDABuffer(d_A, b * m * k);

    // initiate B
    float *d_B = (float *)request->GetArg(1);
    RandCUDABuffer(d_B, b * k * n);

    request->FinishIOBinding();
    fprintf(stderr, "session.io_binding done\n");

    auto status_run = session.Run(*request);
    BRT_TEST_CHECK_STATUS(status_run);
    fprintf(stderr, "session.run done\n");

    auto status_sync = request->Sync();
    BRT_TEST_CHECK_STATUS(status_sync);
    fprintf(stderr, "session.sync done\n");

    float *d_C = (float *)request->GetArg(2);
    CheckBMMPermute(d_A, d_B, d_C, b, m, n, k, d1, 1e-4f);

    // run 2
    RandCUDABuffer(d_A, b * m * k);
    RandCUDABuffer(d_B, b * k * n);

    auto status_run_2 = session.Run(*request);
    BRT_TEST_CHECK_STATUS(status_run_2);

    auto status_sync_2 = request->Sync();
    BRT_TEST_CHECK_STATUS(status_sync_2);
    CheckBMMPermute(d_A, d_B, d_C, b, m, n, k, d1, 1e-4f);
  }
  */
}
