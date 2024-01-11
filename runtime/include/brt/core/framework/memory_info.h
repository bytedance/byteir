// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
// ===========================================================================
// Modification Copyright 2022 ByteDance Ltd. and/or its affiliates.

#pragma once

#include <cstring>
#include <sstream>
#include <string>

namespace brt {
/**
 * memory types for allocator, exec provider specific types should be extended
 * in each provider Whenever this struct is updated, please also update the
 * MakeKey function in brt/core/framework/execution_provider.cc
 */
enum class BrtMemType {
  CPUInput = -2,   // Any CPU memory used by non-CPU execution provider
  CPUOutput = -1,  // CPU accessible memory outputted by non-CPU
                   // execution provider, i.e. CUDA_PINNED
  CPU = CPUOutput, // temporary CPU accessible memory allocated by
                   // non-CPU execution provider, i.e. CUDA_PINNED
  Default = 0,     // the default allocator for execution provider
};

enum class BrtAllocatorType {
  Invalid = -1,
  DeviceAllocator = 0,
  ArenaAllocator = 1,
  CustomAllocator = 2
};

struct BrtMemoryInfo {
  BrtMemoryInfo() = default; // to allow default construction of Tensor

  // use string for name, so we could have customized allocator in execution
  // provider.
  const char *name = nullptr;
  const char *key = nullptr;
  int id = -1;
  BrtMemType mem_type = BrtMemType::Default;
  BrtAllocatorType alloc_type = BrtAllocatorType::Invalid;

  constexpr BrtMemoryInfo(const char *name_, BrtAllocatorType type_,
                          int id_ = 0,
                          BrtMemType mem_type_ = BrtMemType::Default)
#if ((defined(__GNUC__) && __GNUC__ > 4) || defined(__clang__))
      // this causes a spurious error in CentOS gcc 4.8 build so disable if GCC
      // version < 5
      __attribute__((nonnull))
#endif
      : name(name_), key(name_), id(id_), mem_type(mem_type_),
        alloc_type(type_) {
  }

  constexpr BrtMemoryInfo(const char *name_, const char *key_,
                          BrtAllocatorType type_, int id_ = 0,
                          BrtMemType mem_type_ = BrtMemType::Default)
#if ((defined(__GNUC__) && __GNUC__ > 4) || defined(__clang__))
      // this causes a spurious error in CentOS gcc 4.8 build so disable if GCC
      // version < 5
      __attribute__((nonnull))
#endif
      : name(name_), key(key_), id(id_), mem_type(mem_type_),
        alloc_type(type_) {
  }

  // To make OrtMemoryInfo become a valid key in std map
  bool operator<(const BrtMemoryInfo &other) const {
    if (alloc_type != other.alloc_type)
      return alloc_type < other.alloc_type;
    if (mem_type != other.mem_type)
      return mem_type < other.mem_type;
    if (id != other.id)
      return id < other.id;

    return strcmp(name, other.name) < 0;
  }

  std::string ToString() const {
    std::ostringstream ostr;
    ostr << "BrtMemoryInfo:["
         << "name:" << name << " id:" << id
         << " BrtMemType:" << static_cast<int>(mem_type)
         << " BrtAllocatorType:" << static_cast<int>(alloc_type) << "]";
    return ostr.str();
  }
};

inline bool operator==(const BrtMemoryInfo &left, const BrtMemoryInfo &other) {
  return left.mem_type == other.mem_type &&
         left.alloc_type == other.alloc_type && left.id == other.id &&
         strcmp(left.name, other.name) == 0;
}

inline bool operator!=(const BrtMemoryInfo &lhs, const BrtMemoryInfo &rhs) {
  return !(lhs == rhs);
}

std::ostream &operator<<(std::ostream &out, const BrtMemoryInfo &info);
} // namespace brt
