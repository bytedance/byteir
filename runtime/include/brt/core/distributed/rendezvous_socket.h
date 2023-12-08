// Copyright (c) Megvii Inc.
// Licensed under Apache License, Version 2.0
// ===========================================================================
// Modification Copyright 2022 ByteDance Ltd. and/or its affiliates.

#pragma once

#include "brt/core/common/status.h"

#include <mutex>

namespace brt {

int GetFreePort();

common::Status CreateServer(uint32_t nranks, int port);

class RendezvousSocket {
public:
  RendezvousSocket(unsigned nranks, unsigned rank);

  ~RendezvousSocket();

  common::Status connect(const char *master_ip, int port);

  // block until all ranks reach this barrier
  common::Status barrier();

  // the length of send_buff = the length of recv_buff = len
  common::Status broadcast(const void *send_buff, void *recv_buff,
                           size_t send_len, unsigned root);

  // the length of send_buff = sendlen
  // the length of recv_buff = send_len * m_nranks
  common::Status allgather(const void *send_buff, void *recv_buff,
                           size_t send_len);

private:
  uint32_t nranks_;
  uint32_t rank_;
  bool connected_ = false;
  int conn_;
  std::mutex mutex_;
};

} // namespace brt