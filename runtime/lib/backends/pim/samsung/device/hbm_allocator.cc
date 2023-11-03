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



void *HBMAllocator::Alloc(size_t size) {void *p;
  size_t alignment = 32;
    int ret = posix_memalign(&p, alignment, size);
  if (ret != 0)
    abort();

  return p; }

void HBMAllocator::Free(void *p) {  free(p); }

common::Status HBMAllocatorFactory(Session *session, bool use_arena,
                                   size_t size) {

  if (use_arena) {
    auto HBM_allocator = std::make_unique<BFCArena>(
        std::unique_ptr<IAllocator>(new HBMAllocator(0,"HBM")), size);
    auto status = session->AddAllocator(std::move(HBM_allocator));
    return status;
  }

  auto HBM_allocator = std::make_unique<HBMAllocator>(0,"HBM");
  auto status = session->AddAllocator(std::move(HBM_allocator));
  return status;
}

void *HBMExternalAllocator::Alloc(size_t size) {
  void *p = nullptr;
  if (size > 0) {
    p = alloc_(size);
    // review(codemzs): BRT_ENFORCE does not seem appropiate.
    BRT_ENFORCE(p != nullptr);
  }

  return p;
}

void HBMExternalAllocator::Free(void *p) { free_(p); }


// TODO add more option later
common::Status HBMAllocatorFactory(Session *session, int device_id,
                                    bool arena_option, size_t size) {



  

  auto cuda_pinned =
      std::make_unique<HBMAllocator>(device_id, "HBM");
  auto status = session->AddAllocator(std::move(cuda_pinned));

  return status;
}

} // namespace brt
