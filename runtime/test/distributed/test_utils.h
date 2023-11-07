// Copyright (c) Megvii Inc.
// Licensed under Apache License, Version 2.0
// ===========================================================================
// Modification Copyright 2022 ByteDance Ltd. and/or its affiliates.

#pragma once

#include "brt/backends/cuda/device/distributed/distributed_backend_nccl.h"
#include <map>

namespace brt {

struct ContextTrait {
  void *(*alloc)(size_t size);
  void (*set_device)(size_t device);
  void (*free)(void *ptr);
  std::shared_ptr<DContext> (*make_context)();
  void (*sync_context)(std::shared_ptr<DContext> context);
  void (*memcpy_h2d)(void *dst, void *src, size_t len,
                     std::shared_ptr<DContext> context);
  void (*memcpy_d2h)(void *dst, void *src, size_t len,
                     std::shared_ptr<DContext> context);
};

void *alloc_cuda(size_t size);
void set_device_cuda(size_t device);
void free_cuda(void *ptr);
std::shared_ptr<DContext> make_context_cuda();
void sync_context_cuda(std::shared_ptr<DContext> context);
void memcpy_h2d_cuda(void *dst, void *src, size_t len,
                     std::shared_ptr<DContext> context);
void memcpy_d2h_cuda(void *dst, void *src, size_t len,
                     std::shared_ptr<DContext> context);

static std::map<std::string, ContextTrait> context_trait_map = {
    {"BRT_CTX_CUDA",
     {&alloc_cuda, &set_device_cuda, &free_cuda, &make_context_cuda,
      &sync_context_cuda, &memcpy_h2d_cuda, &memcpy_d2h_cuda}}};

typedef enum {
  BRT_NCCL = 0,
} BackendType;

static std::string get_preferred_context(BackendType backend) {
  switch (backend) {
  case BRT_NCCL:
    return "BRT_CTX_CUDA";
  default:
    return "";
  }
}

static ContextTrait get_context_trait(std::string type) {
  return context_trait_map[type];
}

} // namespace brt
