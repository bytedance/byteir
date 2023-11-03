// Copyright (c) Megvii Inc.
// Licensed under Apache License, Version 2.0
// ===========================================================================
// Modification Copyright 2022 ByteDance Ltd. and/or its affiliates.

#include "brt/core/distributed/distributed_backend.h"

using namespace brt::common;

namespace brt {

Status DistributedBackend::init(const char *master_ip, int port) {
  m_client = std::make_shared<RendezvousSocket>(m_nranks, m_rank);
  auto status = m_client->connect(master_ip, port);
  if (status != Status::OK())
    return status;
  return do_init();
}

Status DistributedBackend::init(BcastCallback cb) { return do_init(cb); }

Status DistributedBackend::recv(void *recvbuf, size_t len, DType dtype,
                                uint32_t rank, std::shared_ptr<DContext> ctx) {
  size_t type_size = GetDTypeSize(dtype);
  return _recv(recvbuf, len * type_size, rank, ctx);
}

Status DistributedBackend::send(const void *sendbuff, size_t len, DType dtype,
                                uint32_t rank, std::shared_ptr<DContext> ctx) {
  size_t type_size = GetDTypeSize(dtype);
  return _send(sendbuff, len * type_size, rank, ctx);
}

} // namespace brt
