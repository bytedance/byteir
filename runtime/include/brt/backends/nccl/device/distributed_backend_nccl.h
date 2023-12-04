// Copyright (c) Megvii Inc.
// Licensed under Apache License, Version 2.0
// ===========================================================================
// Modification Copyright 2022 ByteDance Ltd. and/or its affiliates.

#pragma once

#include "brt/core/common/status.h"
#include "brt/core/distributed/distributed_backend.h"
#include "brt/core/framework/dtype.h"

namespace brt {

class DistributedBackendNCCLPrivate;

// Distributed Backend implemented by nccl
// collective communications are performed asynchronously
class DistributedBackendNCCL : public DistributedBackend {
public:
  DistributedBackendNCCL(int nranks, int rank);

  ~DistributedBackendNCCL();

  common::Status do_init() override;
  common::Status do_init(BcastCallback cb) override;

  common::Status _send(const void *sendbuff, size_t size, uint32_t rank,
                       std::shared_ptr<DContext> ctx) override;

  common::Status _recv(void *recvbuff, size_t size, uint32_t rank,
                       std::shared_ptr<DContext> ctx) override;

  common::Status scatter(const void *sendbuff, void *recvbuff, size_t recvlen,
                         DTypeEnum dtype, uint32_t root,
                         std::shared_ptr<DContext> ctx) override;

  common::Status gather(const void *sendbuff, void *recvbuff, size_t sendlen,
                        DTypeEnum dtype, uint32_t root,
                        std::shared_ptr<DContext> ctx) override;

  common::Status all_to_all(const void *sendbuff, void *recvbuff, size_t len,
                            DTypeEnum dtype,
                            std::shared_ptr<DContext> ctx) override;

  common::Status all_gather(const void *sendbuff, void *recvbuff,
                            size_t sendlen, DTypeEnum dtype,
                            std::shared_ptr<DContext> ctx) override;

  common::Status all_reduce(const void *sendbuff, void *recvbuff, size_t len,
                            DTypeEnum dtype, ReduceOp op,
                            std::shared_ptr<DContext> ctx) override;

  common::Status reduce_scatter(const void *sendbuff, void *recvbuff,
                                size_t recvlen, DTypeEnum dtype, ReduceOp op,
                                std::shared_ptr<DContext> ctx) override;

  common::Status broadcast(const void *sendbuff, void *recvbuff, size_t len,
                           DTypeEnum dtype, uint32_t root,
                           std::shared_ptr<DContext> ctx) override;

  common::Status reduce(const void *sendbuff, void *recvbuff, size_t len,
                        DTypeEnum dtype, ReduceOp op, uint32_t root,
                        std::shared_ptr<DContext> ctx) override;

  common::Status group_start() override;
  common::Status group_end() override;

private:
  std::unique_ptr<DistributedBackendNCCLPrivate> m_nccl;
};

} // namespace brt
