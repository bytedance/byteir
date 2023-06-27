//===- rng.h --------------------------------------------------*--- C++ -*-===//
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

#include "./kernels/rng.h"
#include "brt/backends/cuda/device/utils/op_kernel_impl_helpers.h"
#include "brt/core/framework/op_accessor.h"

namespace brt {
namespace cuda {
class RngImplBase {
protected:
  RngImplBase(const OpAccessor &accessor) {
    auto shape = accessor.GetArgShape(0);
    nr_elems = accessor.GetNumElementsOfShape(shape);

    auto dtype = accessor.GetArgDTypeEnum(0);
    BRT_ENFORCE(dtype == DTypeEnum::Float32 || dtype == DTypeEnum::Float64,
                "only float32/64 is supported now");
  }

  size_t nr_elems;
};

template <typename InputTy> class RngUniformImpl;

template <> class RngUniformImpl<float> : public RngImplBase {
public:
  RngUniformImpl(const OpAccessor &accessor) : RngImplBase(accessor) {
    if (accessor.HasAttr("low")) {
      BRT_ENFORCE(accessor.HasAttr("high"));
      low = accessor.GetAttrAsFloat("low");
      high = accessor.GetAttrAsFloat("high");
      BRT_ENFORCE(low < high, "invalid uniform rng attributes");
    } else {
      BRT_ENFORCE(!accessor.HasAttr("high"));
      low = 0;
      high = 1;
    }
  }

  void Execute(float *ptr, curandGenerator_t generator, cudaStream_t stream) {
    if (low == 0 && high == 1) {
      BRT_CURAND_CHECK(curandGenerateUniform(generator, ptr, nr_elems));
    } else {
      kernel::RngUniform<float>(stream, ptr, nr_elems, low, high);
    }
  }

private:
  float low, high;
};

template <> class RngUniformImpl<double> : public RngImplBase {
public:
  RngUniformImpl(const OpAccessor &accessor) : RngImplBase(accessor) {
    if (accessor.HasAttr("low")) {
      BRT_ENFORCE(accessor.HasAttr("high"));
      low = accessor.GetAttrAsFloat("low");
      high = accessor.GetAttrAsFloat("high");
      BRT_ENFORCE(low < high, "invalid uniform rng attributes");
    } else {
      BRT_ENFORCE(!accessor.HasAttr("high"));
      low = 0;
      high = 1;
    }
  }

  void Execute(double *ptr, curandGenerator_t generator, cudaStream_t stream) {
    if (low == 0 && high == 1) {
      BRT_CURAND_CHECK(curandGenerateUniformDouble(generator, ptr, nr_elems));
    } else {
      kernel::RngUniform<double>(stream, ptr, nr_elems, low, high);
    }
  }

private:
  double low, high;
};

class RngNormalImpl : public RngImplBase {
public:
  RngNormalImpl(const OpAccessor &accessor) : RngImplBase(accessor) {
    mean = accessor.GetAttrAsFloat("mean");
    stddev = accessor.GetAttrAsFloat("stddev");
  }

  void Execute(float *ptr, curandGenerator_t generator, cudaStream_t) {
    BRT_CURAND_CHECK(
        curandGenerateNormal(generator, ptr, nr_elems, mean, stddev));
  }

private:
  float mean, stddev;
};

template <typename InputTy>
using RngUniform =
    CurandOpKernel<RngUniformImpl<InputTy>, TypedOperand<InputTy *, 0>>;

using RngNormal = CurandOpKernel<RngNormalImpl, TypedOperand<float *, 0>>;

} // namespace cuda
} // namespace brt
