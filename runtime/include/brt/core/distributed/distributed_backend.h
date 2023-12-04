// Copyright (c) Megvii Inc.
// Licensed under Apache License, Version 2.0
// ===========================================================================
// Modification Copyright 2022 ByteDance Ltd. and/or its affiliates.

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "brt/core/common/enums.h"
#include "brt/core/distributed/d_context.h"
#include "brt/core/distributed/rendezvous_socket.h"
#include "brt/core/framework/dtype.h"

namespace brt {

using BcastCallback = std::function<void(char *, size_t len)>;

class DistributedBackend {
public:
  DistributedBackend(uint32_t nranks, uint32_t rank)
      : m_nranks(nranks), m_rank(rank) {}

  // get the number of all ranks
  uint32_t nranks() { return m_nranks; }

  // get the rank of this process
  uint32_t rank() { return m_rank; }

  // establish connection with server
  common::Status init(const char *master_ip, int port);
  common::Status init(BcastCallback cb);

  // send data to another communicator in the group
  // implemented in the subclass _send()
  common::Status send(const void *sendbuff, size_t len, DTypeEnum dtype,
                      uint32_t rank, std::shared_ptr<DContext> ctx);

  // receive data from another communicator in the group
  // implemented in the subclass _recv()
  common::Status recv(void *recvbuf, size_t len, DTypeEnum dtype, uint32_t rank,
                      std::shared_ptr<DContext> ctx);

  // implemented in the subclass and called in init()
  virtual common::Status do_init() = 0;
  virtual common::Status do_init(BcastCallback cb) = 0;

  // TODO: enhance api to support arbitary device groups
  // the length of sendbuff = recvlen * m_nranks
  // the length of recvbuff = recvlen
  virtual common::Status scatter(const void *sendbuff, void *recvbuff,
                                 size_t recvlen, DTypeEnum dtype, uint32_t root,
                                 std::shared_ptr<DContext> ctx) = 0;

  // TODO: enhance api to support arbitary device groups
  // the length of sendbuff = sendlen
  // the length of recvbuff = sendlen * m_nranks
  virtual common::Status gather(const void *sendbuff, void *recvbuff,
                                size_t sendlen, DTypeEnum dtype, uint32_t root,
                                std::shared_ptr<DContext> ctx) = 0;

  // TODO: enhance api to support arbitary device groups
  // the length of sendbuff = the length of recvbuff = len * m_nranks
  virtual common::Status all_to_all(const void *sendbuff, void *recvbuff,
                                    size_t len, DTypeEnum dtype,
                                    std::shared_ptr<DContext> ctx) = 0;

  // TODO: enhance api to support arbitary device groups
  // the length of sendbuff = sendlen
  // the length of recvbuff = sendlen * m_nranks
  virtual common::Status all_gather(const void *sendbuff, void *recvbuff,
                                    size_t sendlen, DTypeEnum dtype,
                                    std::shared_ptr<DContext> ctx) = 0;
  // TODO: enhance api to support arbitary device groups
  // the length of sendbuff = the length of recvbuff = len
  virtual common::Status all_reduce(const void *sendbuff, void *recvbuff,
                                    size_t len, DTypeEnum dtype, ReduceOp op,
                                    std::shared_ptr<DContext> ctx) = 0;

  // TODO: enhance api to support arbitary device groups
  // the length of sendbuff = recvlen * m_nranks
  // the length of recvbuff = recvlen
  virtual common::Status reduce_scatter(const void *sendbuff, void *recvbuff,
                                        size_t recvlen, DTypeEnum dtype,
                                        ReduceOp op,
                                        std::shared_ptr<DContext> ctx) = 0;

  // TODO: enhance api to support arbitary device groups
  // the length of sendbuff = the length of recvbuff = len
  virtual common::Status broadcast(const void *sendbuff, void *recvbuff,
                                   size_t len, DTypeEnum dtype, uint32_t root,
                                   std::shared_ptr<DContext> ctx) = 0;

  // TODO: enhance api to support arbitary device groups
  // the length of sendbuff = the length of recvbuff = len
  virtual common::Status reduce(const void *sendbuff, void *recvbuff,
                                size_t len, DTypeEnum dtype, ReduceOp op,
                                uint32_t root,
                                std::shared_ptr<DContext> ctx) = 0;

  // mark the begin of a series of (send recv)
  virtual common::Status group_start() = 0;
  // mark the end of a series of (send recv)
  virtual common::Status group_end() = 0;

protected:
  uint32_t m_nranks;
  uint32_t m_rank;
  std::shared_ptr<RendezvousSocket> m_client;

  // send data to another communicator in the group
  virtual common::Status _send(const void *sendbuff, size_t size, uint32_t rank,
                               std::shared_ptr<DContext> ctx) = 0;

  // receive data from another communicator in the group
  virtual common::Status _recv(void *recvbuf, size_t size, uint32_t rank,
                               std::shared_ptr<DContext> ctx) = 0;
};

} // namespace brt
