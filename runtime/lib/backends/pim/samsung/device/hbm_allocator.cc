// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
// ===========================================================================
// Modification Copyright 2022 ByteDance Ltd. and/or its affiliates.

#include "brt/backends/pim/samsung/device/hbm_allocator.h"
#include "brt/core/common/common.h"
#include "brt/core/framework/arena.h"
#include "brt/core/framework/bfc_arena.h"
#include "brt/core/session/session.h"

using namespace brt;
using namespace brt::common;

namespace brt {

void *HBMPIMAllocator::Alloc(size_t size) {
  void *p;
  size_t alignment = 32;
  int ret = posix_memalign(&p, alignment, size);
  if (ret != 0)
    abort();

  return p;
}

void HBMPIMoutputAllocator::Free(void *p) { free(p); }

void *HBMPIMoutputAllocator::Alloc(size_t size) {
  void *p;
  size_t alignment = 32;
  int ret = posix_memalign(&p, alignment, size);
  if (ret != 0)
    abort();

  return p;
}

void HBMPIMAllocator::Free(void *p) { free(p); }
common::Status HBMPIMAllocatorFactory(Session *session, bool use_arena,
                                      size_t size) {

  if (use_arena) {
    auto HBMPIM_allocator = std::make_unique<BFCArena>(
        std::unique_ptr<IAllocator>(new HBMPIMAllocator(0, "HBMPIM")), size);
    auto status = session->AddAllocator(std::move(HBMPIM_allocator));
    return status;
  }

  auto HBMPIM_allocator = std::make_unique<HBMPIMAllocator>(0, "HBMPIM");
  auto status = session->AddAllocator(std::move(HBMPIM_allocator));
  return status;
}

void *HBMPIMExternalAllocator::Alloc(size_t size) {
  void *p = nullptr;
  if (size > 0) {
    p = alloc_(size);
    // review(codemzs): BRT_ENFORCE does not seem appropiate.
    BRT_ENFORCE(p != nullptr);
  }

  return p;
}

void HBMPIMExternalAllocator::Free(void *p) { free_(p); }

// TODO add more option later
common::Status HBMPIMAllocatorFactory(Session *session, int device_id,
                                      bool arena_option, size_t size) {

  // if (arena_option) {
  //   auto hbmpim = std::make_unique<BFCArena>(
  //       std::unique_ptr<IAllocator>(new HBMPIMAllocator(device_id, "hbmpim")),
  //       size);
  //   auto status = session->AddAllocator(std::move(hbmpim));
  //   if (!status.IsOK())
  //     return status;
  // } else {
    auto hbmpim = std::make_unique<HBMPIMAllocator>(device_id, "HBMPIM");
    auto status = session->AddAllocator(std::move(hbmpim));
    if (!status.IsOK())
      return status;
  // }

//   auto hbm_output =
//       std::make_unique<HBMPIMAllocator>(device_id, "hbm_output");
// return session->AddAllocator(std::move(hbm_output));
}
} // namespace brt
