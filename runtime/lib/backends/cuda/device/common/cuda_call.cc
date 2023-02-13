// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// ===========================================================================
// Modification Copyright 2022 ByteDance Ltd. and/or its affiliates.

#include "brt/backends/cuda/device/common/cuda_call.h"

#include "brt/core/common/common.h"
#include "brt/core/common/logging/logging.h"
#include "brt/core/common/logging/macros.h"

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4505)
#endif

#include "cutlass/cutlass.h"
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cufft.h>
#include <curand.h>
#include <cusparse.h>
#include <memory>
#include <nvrtc.h>

#ifdef BRT_USE_NCCL
#include <nccl.h>
#endif

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#ifdef _WIN32
#else // POSIX
#include <string.h>
#include <unistd.h>
#endif

using namespace brt::common;
using namespace brt::logging;

namespace brt {
namespace cuda {

template <typename ERRTYPE> const char *CudaErrString(ERRTYPE) {
  BRT_NOT_IMPLEMENTED();
}

#define CASE_ENUM_TO_STR(x)                                                    \
  case x:                                                                      \
    return #x

template <> const char *CudaErrString<CUresult>(CUresult x) {
  cudaDeviceSynchronize();
  const char *msg;
  cuGetErrorName(x, &msg);
  return msg;
}

template <> const char *CudaErrString<nvrtcResult>(nvrtcResult x) {
  return nvrtcGetErrorString(x);
}

template <> const char *CudaErrString<cudaError_t>(cudaError_t x) {
  cudaDeviceSynchronize();
  return cudaGetErrorString(x);
}

template <> const char *CudaErrString<cublasStatus_t>(cublasStatus_t e) {
  cudaDeviceSynchronize();

  switch (e) {
    CASE_ENUM_TO_STR(CUBLAS_STATUS_SUCCESS);
    CASE_ENUM_TO_STR(CUBLAS_STATUS_NOT_INITIALIZED);
    CASE_ENUM_TO_STR(CUBLAS_STATUS_ALLOC_FAILED);
    CASE_ENUM_TO_STR(CUBLAS_STATUS_INVALID_VALUE);
    CASE_ENUM_TO_STR(CUBLAS_STATUS_ARCH_MISMATCH);
    CASE_ENUM_TO_STR(CUBLAS_STATUS_MAPPING_ERROR);
    CASE_ENUM_TO_STR(CUBLAS_STATUS_EXECUTION_FAILED);
    CASE_ENUM_TO_STR(CUBLAS_STATUS_INTERNAL_ERROR);
    CASE_ENUM_TO_STR(CUBLAS_STATUS_NOT_SUPPORTED);
    CASE_ENUM_TO_STR(CUBLAS_STATUS_LICENSE_ERROR);
  default:
    return "(look for CUBLAS_STATUS_xxx in cublas_api.h)";
  }
}

template <> const char *CudaErrString<curandStatus>(curandStatus) {
  cudaDeviceSynchronize();
  return "(see curand.h & look for curandStatus or CURAND_STATUS_xxx)";
}

template <> const char *CudaErrString<cudnnStatus_t>(cudnnStatus_t e) {
  cudaDeviceSynchronize();
  return cudnnGetErrorString(e);
}

template <> const char *CudaErrString<cufftResult>(cufftResult e) {
  cudaDeviceSynchronize();
  switch (e) {
    CASE_ENUM_TO_STR(CUFFT_SUCCESS);
    CASE_ENUM_TO_STR(CUFFT_ALLOC_FAILED);
    CASE_ENUM_TO_STR(CUFFT_INVALID_VALUE);
    CASE_ENUM_TO_STR(CUFFT_INTERNAL_ERROR);
    CASE_ENUM_TO_STR(CUFFT_SETUP_FAILED);
    CASE_ENUM_TO_STR(CUFFT_INVALID_SIZE);
  default:
    return "Unknown cufft error status";
  }
}

template <> const char *CudaErrString<cutlass::Status>(cutlass::Status e) {
  cudaDeviceSynchronize();
  return cutlassGetStatusString(e);
}

#ifdef BRT_USE_NCCL
template <> const char *CudaErrString<ncclResult_t>(ncclResult_t e) {
  cudaDeviceSynchronize();
  return ncclGetErrorString(e);
}
#endif

template <typename ERRTYPE, bool THRW>
[[nodiscard]] std::conditional_t<THRW, void, common::Status>
CudaCall(ERRTYPE retCode, const char *exprString, const char *libName,
         ERRTYPE successCode, const char *msg) {
  if (retCode != successCode) {
    try {
#ifdef _WIN32
      auto del = [](char *p) { free(p); };
      std::unique_ptr<char, decltype(del)> hostname_ptr(nullptr, del);
      size_t hostname_len = 0;
      char *hostname = nullptr;
      // TODO: avoid using const_cast
      if (-1 == _dupenv_s(&hostname, &hostname_len, "COMPUTERNAME"))
        hostname = const_cast<char *>("?");
      else
        hostname_ptr.reset(hostname);
#else
      char hostname[HOST_NAME_MAX];
      if (gethostname(hostname, HOST_NAME_MAX) != 0)
        strcpy(hostname, "?");
#endif
      int currentCudaDevice;
      cudaGetDevice(&currentCudaDevice);
      cudaGetLastError(); // clear last CUDA error
      static char str[1024];
      snprintf(str, 1024,
               "%s failure %d: %s ; GPU=%d ; hostname=%s ; expr=%s; %s",
               libName, (int)retCode, CudaErrString(retCode), currentCudaDevice,
               hostname, exprString, msg);
      if constexpr (THRW) {
        // throw an exception with the error info
        BRT_THROW(str);
      } else {
        return Status(StatusCategory::BRT, StatusCode::FAIL, str);
      }
    } catch (const std::exception &
                 e) { // catch, log, and rethrow since CUDA code sometimes hangs
                      // in destruction, so we'd never get to see the error
      if constexpr (THRW) {
        BRT_THROW(e.what());
      } else {
        return Status(StatusCategory::BRT, StatusCode::FAIL, e.what());
      }
    }
  }
  if constexpr (!THRW) {
    return Status::OK();
  }
}

template Status CudaCall<CUresult, false>(CUresult retCode,
                                          const char *exprString,
                                          const char *libName,
                                          CUresult successCode,
                                          const char *msg);
template void CudaCall<CUresult, true>(CUresult retCode, const char *exprString,
                                       const char *libName,
                                       CUresult successCode, const char *msg);
template Status CudaCall<cudaError, false>(cudaError retCode,
                                           const char *exprString,
                                           const char *libName,
                                           cudaError successCode,
                                           const char *msg);
template void CudaCall<cudaError, true>(cudaError retCode,
                                        const char *exprString,
                                        const char *libName,
                                        cudaError successCode, const char *msg);
template Status CudaCall<cublasStatus_t, false>(cublasStatus_t retCode,
                                                const char *exprString,
                                                const char *libName,
                                                cublasStatus_t successCode,
                                                const char *msg);
template void CudaCall<cublasStatus_t, true>(cublasStatus_t retCode,
                                             const char *exprString,
                                             const char *libName,
                                             cublasStatus_t successCode,
                                             const char *msg);
template Status CudaCall<cudnnStatus_t, false>(cudnnStatus_t retCode,
                                               const char *exprString,
                                               const char *libName,
                                               cudnnStatus_t successCode,
                                               const char *msg);
template void CudaCall<cudnnStatus_t, true>(cudnnStatus_t retCode,
                                            const char *exprString,
                                            const char *libName,
                                            cudnnStatus_t successCode,
                                            const char *msg);
template Status CudaCall<curandStatus_t, false>(curandStatus_t retCode,
                                                const char *exprString,
                                                const char *libName,
                                                curandStatus_t successCode,
                                                const char *msg);
template void CudaCall<curandStatus_t, true>(curandStatus_t retCode,
                                             const char *exprString,
                                             const char *libName,
                                             curandStatus_t successCode,
                                             const char *msg);
template Status CudaCall<cufftResult, false>(cufftResult retCode,
                                             const char *exprString,
                                             const char *libName,
                                             cufftResult successCode,
                                             const char *msg);
template void CudaCall<cufftResult, true>(cufftResult retCode,
                                          const char *exprString,
                                          const char *libName,
                                          cufftResult successCode,
                                          const char *msg);
template Status CudaCall<cutlass::Status, false>(cutlass::Status retCode,
                                                 const char *exprString,
                                                 const char *libName,
                                                 cutlass::Status successCode,
                                                 const char *msg);
template void CudaCall<cutlass::Status, true>(cutlass::Status retCode,
                                              const char *exprString,
                                              const char *libName,
                                              cutlass::Status successCode,
                                              const char *msg);

// TODO add a build a flag for nvrtc
template Status CudaCall<nvrtcResult, false>(nvrtcResult retCode,
                                             const char *exprString,
                                             const char *libName,
                                             nvrtcResult successCode,
                                             const char *msg);
template void CudaCall<nvrtcResult, true>(nvrtcResult retCode,
                                          const char *exprString,
                                          const char *libName,
                                          nvrtcResult successCode,
                                          const char *msg);

#ifdef BRT_USE_NCCL
template Status CudaCall<ncclResult_t, false>(ncclResult_t retCode,
                                              const char *exprString,
                                              const char *libName,
                                              ncclResult_t successCode,
                                              const char *msg);
#endif

} // namespace cuda
} // namespace brt
