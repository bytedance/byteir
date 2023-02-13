//===- cuda_env.h ---------------------------------------------*--- C++ -*-===//
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

struct CUctx_st;
struct CUstream_st;

namespace brt {
namespace cuda {
class CudaEnv {
public:
  CudaEnv(int device_id);
  CudaEnv(CUctx_st *ctx);
  CudaEnv(CUstream_st *stream);

  void Activate();

  // return true if this CudaEnv is associated with the cuda primary context
  bool IsPrimaryContext() { return is_primary_; }

  int GetDeviceID() { return device_id_; }

private:
  void Initialize(CUctx_st *st);

  int device_id_;
  bool is_primary_;
  CUctx_st *ctx_;
};
} // namespace cuda
} // namespace brt
