//===- engine_util.h ------------------------------------------*--- C++ -*-===//
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

#include <vector>

/**
 * This file holds utility functions for execution engine for generated function
 */

namespace brt {
template <typename T, size_t N> struct MLIRStridedMemRefType {
  T *basePtr;
  T *data;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};
template <typename T> struct MLIRStridedMemRefType<T, 0> {
  T *basePtr;
  T *data;
  int64_t offset;
};

template <typename T> struct MLIRUnrankedMemRefType {
  int64_t rank;
  void *descriptor;
};

template <typename T> class MLIRDynamicMemRefType {
public:
  explicit MLIRDynamicMemRefType(const MLIRUnrankedMemRefType<T> &memRef)
      : rank(memRef.rank) {
    auto *desc =
        static_cast<MLIRStridedMemRefType<char, 1> *>(memRef.descriptor);
    basePtr = desc->basePtr;
    data = desc->data;
    offset = desc->offset;
    sizes = rank == 0 ? nullptr : desc->sizes;
    strides = sizes + rank;
  }

  int64_t rank;
  T *basePtr;
  T *data;
  int64_t offset;
  const int64_t *sizes;
  const int64_t *strides;
};

struct MLIREngineMemRefDescriptor {
  static constexpr size_t kMaxNumDimensions = 8;

  void *data;
  void *aligned_data;
  uint64_t offset;
  std::vector<uint64_t> shape;
  std::vector<uint64_t> stride;
  std::aligned_storage_t<sizeof(MLIRStridedMemRefType<void, kMaxNumDimensions>),
                         alignof(
                             MLIRStridedMemRefType<void, kMaxNumDimensions>)>
      storage;

  // default
  MLIREngineMemRefDescriptor()
      : data(nullptr), aligned_data(nullptr), offset(0) {}

  // TODO: add more constructor later
  MLIREngineMemRefDescriptor(void *ptr, size_t rank)
      : data(ptr), aligned_data(ptr), offset(0) {
    shape.resize(rank, 0);
    stride.resize(rank, 0);
  }

  MLIREngineMemRefDescriptor(void *ptr, const std::vector<int64_t> &shape_)
      : data(ptr), aligned_data(ptr), offset(0) {
    shape.insert(shape.end(), shape_.begin(), shape_.end());
    // initialize as contiguous strides
    int ndim = shape.size();
    stride.resize(ndim);
    uint64_t cur = 1;
    for (size_t i = ndim; i; --i) {
      stride[i - 1] = cur;
      cur = cur * shape[i - 1];
    }
  }

  void Update(void *ptr) {
    data = ptr;
    aligned_data = ptr;
  }

  void *GetMemrefPtr() {
    switch (shape.size()) {
#define Case(N)                                                                \
  case N:                                                                      \
    CreateStridedMemRefImpl(std::make_index_sequence<N>{});                    \
    break;
      Case(0);
      Case(1);
      Case(2);
      Case(3);
      Case(4);
      Case(5);
      Case(6);
      Case(7);
    }
    return &storage;
  }
#undef Case
private:
  template <size_t... Idx>
  void CreateStridedMemRefImpl(std::index_sequence<Idx...>) {
    new (&storage) MLIRStridedMemRefType<void, sizeof...(Idx)>{
        data, aligned_data, static_cast<int64_t>(offset),
        static_cast<int64_t>(shape[Idx])...,
        static_cast<int64_t>(stride[Idx])...};
  }
};

inline void InsertMemDescToArgs(MLIREngineMemRefDescriptor &desc,
                                std::vector<void *> &args) {
  args.push_back(&desc.data);
  args.push_back(&desc.aligned_data);
  args.push_back(&desc.offset);

  size_t rank = desc.shape.size();
  for (size_t i = 0; i < rank; ++i) {
    args.push_back(&desc.shape[i]);
  }

  for (size_t i = 0; i < rank; ++i) {
    args.push_back(&desc.stride[i]);
  }
}

} // namespace brt
