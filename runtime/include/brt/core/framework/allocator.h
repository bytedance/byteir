// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
// ===========================================================================
// Modification Copyright 2022 ByteDance Ltd. and/or its affiliates.

#pragma once

#include "brt/core/common/status.h"
#include "memory_info.h"
#include <functional>
#include <memory>

// This configures the arena based allocator used by ORT
// See docs/C_API.md for details on what these mean and how to choose these
// values
struct BrtArenaCfg {
  BrtArenaCfg()
      : max_mem(0), arena_extend_strategy(-1), initial_chunk_size_bytes(-1),
        max_dead_bytes_per_chunk(-1), initial_growth_chunk_size_bytes(-1) {}
  BrtArenaCfg(size_t max_mem, int arena_extend_strategy,
              int initial_chunk_size_bytes, int max_dead_bytes_per_chunk,
              int initial_growth_chunk_size_bytes)
      : max_mem(max_mem), arena_extend_strategy(arena_extend_strategy),
        initial_chunk_size_bytes(initial_chunk_size_bytes),
        max_dead_bytes_per_chunk(max_dead_bytes_per_chunk),
        initial_growth_chunk_size_bytes(initial_growth_chunk_size_bytes) {}

  size_t max_mem;            // use 0 to allow BRT to choose the default
  int arena_extend_strategy; // use -1 to allow BRT to choose the default, 0 =
                             // kNextPowerOfTwo, 1 = kSameAsRequested
  int initial_chunk_size_bytes; // use -1 to allow BRT to choose the default
  int max_dead_bytes_per_chunk; // use -1 to allow BRT to choose the default
  int initial_growth_chunk_size_bytes; // use -1 to allow BRT to choose the
                                       // default
};

namespace brt {
class Session;

constexpr const char *CPU = "Cpu";
constexpr const char *CUDA = "Cuda";
constexpr const char *CUDA_PINNED = "CudaPinned";

constexpr size_t kAllocAlignment = 256;

template <typename T>
using IAllocatorUniquePtr = std::unique_ptr<T, std::function<void(T *)>>;

class IAllocator {
public:
  IAllocator(const BrtMemoryInfo &info) : memory_info_(info) {}
  virtual ~IAllocator() = default;
  /**
  @remarks Use SafeInt when calculating the size of memory to allocate using
  Alloc.
  @remarks LWC: disable SafeInt for now
  */
  virtual void *Alloc(size_t size) = 0;
  virtual void Free(void *p) = 0;
  const BrtMemoryInfo &Info() const { return memory_info_; };

  virtual void SetDevice(bool /*throw_when_fail*/) const {}

  static bool CalcMemSizeForArray(size_t nmemb, size_t size,
                                  size_t *out) noexcept {
    return CalcMemSizeForArrayWithAlignment(nmemb, size, 0, out);
  }

  /**
   * Calculate the memory size for an array. The size is bounds checked using
   * SafeInt.  LWC: disable SafeInt for now \tparam alignment must be power of 2
   * \param nmemb Number of members or elements in the array
   * \param size Size of each element
   * \param out Total size required after any alignment is applied
   * \return true, successful. false, overflow
   */
  static bool CalcMemSizeForArrayWithAlignment(size_t nmemb, size_t size,
                                               size_t alignment,
                                               size_t *out) noexcept;

  /**
   * https://cwe.mitre.org/data/definitions/190.html
   * \param alignment must be power of 2
   * \param nmemb Number of members or elements in the array
   * \param size Size of each element
   * \param out Total size required after any alignment is applied
   * \return true, successful. false, overflow
   * \remarks This was the original API and was implemented in the header.
   * Replaced with the above version implemented in the .cc file so that the
   * SafeInt dependency is internal.
   */
  template <size_t alignment>
  static bool CalcMemSizeForArrayWithAlignment(size_t nmemb, size_t size,
                                               size_t *out) noexcept;

  /**
   * allocate memory for an array which has nmemb items of data, each size bytes
   * long
   */
  void *AllocArray(size_t nmemb, size_t size) {
    size_t len;
    if (!CalcMemSizeForArray(nmemb, size, &len))
      return nullptr;
    return Alloc(len);
  }

  /**
   * allocate memory for an array which has nmemb items of data, each size bytes
   * long
   */
  template <size_t alignment>
  void *AllocArrayWithAlignment(size_t nmemb, size_t size) {
    size_t len;
    if (!CalcMemSizeForArrayWithAlignment(nmemb, size, alignment, &len))
      return nullptr;
    return Alloc(len);
  }

  /**
     Create a std::unique_ptr that is allocated and freed by the provided
     IAllocator.
     @param allocator The allocator.
     @param count_or_bytes The exact bytes to allocate if T is void, otherwise
     the number of elements to allocate.
     @returns std::unique_ptr with allocated memory and deleter.
  */
  template <typename T>
  static IAllocatorUniquePtr<T>
  MakeUniquePtr(std::shared_ptr<IAllocator> allocator, size_t count_or_bytes) {
    if (allocator == nullptr)
      return nullptr;
    // for now limit to fundamental types. we could support others, but to do so
    // either we or the caller needs to call the dtor for the objects, for
    // buffers allocated on device we don't have destructor
    // static_assert(std::is_fundamental<T>::value, "Fundamental type required
    // as no destructors are called.");

    size_t alloc_size = count_or_bytes;

    // if T is not void, 'count_or_bytes' == number of items so allow for that
    if (!std::is_void<T>::value) {
      // sizeof(void) isn't valid, but the compiler isn't smart enough to ignore
      // that this line isn't reachable if T is void. use std::conditional to
      // 'use' void* in the sizeof call
      if (!CalcMemSizeForArray(
              count_or_bytes,
              sizeof(typename std::conditional<std::is_void<T>::value, void *,
                                               T>::type),
              &alloc_size))
        return nullptr;
    }

    return IAllocatorUniquePtr<T>{
        static_cast<T *>(allocator->Alloc(alloc_size)), // allocate
        [=](T *ptr) { // capture 'allocator' by value so it's always valid
          allocator->Free(ptr);
        }};
  }

private:
  BrtMemoryInfo memory_info_;
};

template <size_t alignment>
bool IAllocator::CalcMemSizeForArrayWithAlignment(size_t nmemb, size_t size,
                                                  size_t *out) noexcept {
  return CalcMemSizeForArrayWithAlignment(nmemb, size, alignment, out);
}

class CPUAllocator : public IAllocator {
public:
  explicit CPUAllocator(const BrtMemoryInfo &memory_info)
      : IAllocator(memory_info) {}

  CPUAllocator()
      : IAllocator(
            BrtMemoryInfo(CPU, "cpu", BrtAllocatorType::DeviceAllocator)) {}

  void *Alloc(size_t size) override;
  void Free(void *p) override;
};

common::Status CPUAllocatorFactory(Session *session, bool use_arena = false,
                                   size_t size = 1 << 30);

using TAllocator = CPUAllocator;

using AllocatorPtr = std::shared_ptr<IAllocator>;

} // namespace brt
