// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// ===========================================================================
// Modification Copyright 2022 ByteDance Ltd. and/or its affiliates.

#pragma once

#include "brt/core/common/status.h"

namespace brt {
namespace cuda {

// -----------------------------------------------------------------------
// Error handling
// -----------------------------------------------------------------------

template <typename ERRTYPE, bool THRW>
[[nodiscard]] std::conditional_t<THRW, void, common::Status>
CudaCall(ERRTYPE retCode, const char *exprString, const char *libName,
         ERRTYPE successCode, const char *msg = "");

} // namespace cuda
} // namespace brt

#define BRT_CU_CALL(expr)                                                      \
  (::brt::cuda::CudaCall<CUresult, false>((expr), #expr, "CU", CUDA_SUCCESS))
#define BRT_CUDA_CALL(expr)                                                    \
  (::brt::cuda::CudaCall<cudaError, false>((expr), #expr, "CUDA", cudaSuccess))
#define BRT_CUBLAS_CALL(expr)                                                  \
  (::brt::cuda::CudaCall<cublasStatus_t, false>((expr), #expr, "CUBLAS",       \
                                                CUBLAS_STATUS_SUCCESS))
#define BRT_CUSPARSE_CALL(expr)                                                \
  (::brt::cuda::CudaCall<cusparseStatus_t, false>((expr), #expr, "CUSPARSE",   \
                                                  CUSPARSE_STATUS_SUCCESS))
#define BRT_CURAND_CALL(expr)                                                  \
  (::brt::cuda::CudaCall<curandStatus_t, false>((expr), #expr, "CURAND",       \
                                                CURAND_STATUS_SUCCESS))
#define BRT_CUDNN_CALL(expr)                                                   \
  (::brt::cuda::CudaCall<cudnnStatus_t, false>((expr), #expr, "CUDNN",         \
                                               CUDNN_STATUS_SUCCESS))
#define BRT_CUDNN_CALL2(expr, m)                                               \
  (::brt::cuda::CudaCall<cudnnStatus_t, false>((expr), #expr, "CUDNN",         \
                                               CUDNN_STATUS_SUCCESS, m))
#define BRT_CUFFT_CALL(expr)                                                   \
  (::brt::cuda::CudaCall<cufftResult, false>((expr), #expr, "CUFFT",           \
                                             CUFFT_SUCCESS))
#define BRT_CUTLASS_CALL(expr)                                                 \
  (::brt::cuda::CudaCall<cutlass::Status, false>((expr), #expr, "CUTLASS",     \
                                                 cutlass::Status::kSuccess))

#define BRT_CU_CHECK(expr)                                                     \
  (::brt::cuda::CudaCall<CUresult, true>((expr), #expr, "CU", CUDA_SUCCESS))
#define BRT_CUDA_CHECK(expr)                                                   \
  (::brt::cuda::CudaCall<cudaError, true>((expr), #expr, "CUDA", cudaSuccess))
#define BRT_CUBLAS_CHECK(expr)                                                 \
  (::brt::cuda::CudaCall<cublasStatus_t, true>((expr), #expr, "CUBLAS",        \
                                               CUBLAS_STATUS_SUCCESS))
#define BRT_CUSPARSE_CHECK(expr)                                               \
  (::brt::cuda::CudaCall<cusparseStatus_t, true>((expr), #expr, "CUSPARSE",    \
                                                 CUSPARSE_STATUS_SUCCESS))
#define BRT_CURAND_CHECK(expr)                                                 \
  (::brt::cuda::CudaCall<curandStatus_t, true>((expr), #expr, "CURAND",        \
                                               CURAND_STATUS_SUCCESS))
#define BRT_CUDNN_CHECK(expr)                                                  \
  (::brt::cuda::CudaCall<cudnnStatus_t, true>((expr), #expr, "CUDNN",          \
                                              CUDNN_STATUS_SUCCESS))
#define BRT_CUDNN_CHECK2(expr, m)                                              \
  (::brt::cuda::CudaCall<cudnnStatus_t, true>((expr), #expr, "CUDNN",          \
                                              CUDNN_STATUS_SUCCESS, m))
#define BRT_CUFFT_CHECK(expr)                                                  \
  (::brt::cuda::CudaCall<cufftResult, true>((expr), #expr, "CUFFT",            \
                                            CUFFT_SUCCESS))
#define BRT_CUTLASS_CHECK(expr)                                                \
  (::brt::cuda::CudaCall<cutlass::Status, true>((expr), #expr, "CUTLASS",      \
                                                cutlass::Status::kSuccess))

// TODO add a flag for NVRTC
#define BRT_NVRTC_CALL(expr)                                                   \
  (::brt::cuda::CudaCall<nvrtcResult, false>((expr), #expr, "NVRTC",           \
                                             NVRTC_SUCCESS))
#define BRT_NVRTC_CHECK(expr)                                                  \
  (::brt::cuda::CudaCall<nvrtcResult, true>((expr), #expr, "NVRTC",            \
                                            NVRTC_SUCCESS))

#ifdef BRT_USE_NCCL
#define BRT_NCCL_CALL(expr)                                                    \
  (::brt::cuda::CudaCall<ncclResult_t, false>((expr), #expr, "NCCL",           \
                                              ncclSuccess))
#define BRT_NCCL_CHECK(expr)                                                   \
  (::brt::cuda::CudaCall<ncclResult_t, true>((expr), #expr, "NCCL",            \
                                             ncclSuccess))
#endif
