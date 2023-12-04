// Copyright (c) Megvii Inc.
// Licensed under Apache License, Version 2.0
// ===========================================================================
// Modification Copyright 2022 ByteDance Ltd. and/or its affiliates.

#include "brt/core/distributed/rendezvous_socket.h"
#include "brt/core/common/logging/logging.h"
#include "brt/core/common/logging/macros.h"

#include <arpa/inet.h>
#include <cassert>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <string.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

using namespace brt::logging;
using namespace brt::common;

namespace brt {

//===----------------------------------------------------------------------===//
// GetFreePort
//===----------------------------------------------------------------------===//

int GetFreePort() {
  // create socket
  int sock = socket(AF_INET, SOCK_STREAM, 0);
  assert(sock != -1);

  // set address
  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(0);

  // bind
  assert(bind(sock, (struct sockaddr *)&addr, sizeof(addr)) != -1);

  // get port
  socklen_t len = sizeof(addr);
  assert(getsockname(sock, (struct sockaddr *)&addr, &len) != -1);
  int port = ntohs(addr.sin_port);

  // close
  assert(close(sock) != -1);

  return port;
}

//===----------------------------------------------------------------------===//
// CreateServer
//===----------------------------------------------------------------------===//

namespace {

void serve_barrier(uint32_t nranks, int *conns) {
  uint32_t request_id;

  // recv other requests
  for (uint32_t rank = 1; rank < nranks; rank++) {
    assert(recv(conns[rank], &request_id, sizeof(uint32_t), MSG_WAITALL) != -1);
    assert(request_id == 1);
  }

  // send ack
  uint32_t ack = 0;
  for (uint32_t rank = 0; rank < nranks; rank++) {
    assert(send(conns[rank], &ack, sizeof(uint32_t), 0) != -1);
  }
}

void serve_broadcast(uint32_t nranks, int *conns) {
  uint32_t request_id, root, root0;
  uint64_t len, len0;

  // recv request 0
  assert(recv(conns[0], &root0, sizeof(uint32_t), MSG_WAITALL) != -1);
  assert(recv(conns[0], &len0, sizeof(uint64_t), MSG_WAITALL) != -1);

  // recv other requests
  for (uint32_t rank = 1; rank < nranks; rank++) {
    assert(recv(conns[rank], &request_id, sizeof(uint32_t), MSG_WAITALL) != -1);
    assert(request_id == 2 && "inconsistent request_id from rank");

    assert(recv(conns[rank], &root, sizeof(uint32_t), MSG_WAITALL) != -1);
    assert(root == root0 && "inconsistent root from rank");

    assert(recv(conns[rank], &len, sizeof(uint64_t), MSG_WAITALL) != -1);
    assert(len == len0 && "inconsistent len from rank");
  }

  root = root0;
  len = len0;

  // recv data from root
  void *data = malloc(len);
  assert(recv(conns[root], data, len, MSG_WAITALL) != -1);

  // send data to clients
  for (uint32_t rank = 0; rank < nranks; rank++) {
    assert(send(conns[rank], data, len, 0) != -1);
  }

  free(data);
}

void serve_allgather(uint32_t nranks, int *conns) {
  uint32_t request_id;
  uint64_t len, len0;

  // recv request 0
  assert(recv(conns[0], &len0, sizeof(uint64_t), MSG_WAITALL) != -1);

  // recv other requests
  for (uint32_t rank = 1; rank < nranks; rank++) {
    assert(recv(conns[rank], &request_id, sizeof(uint32_t), MSG_WAITALL) != -1);
    assert(request_id == 3 && "inconsistent request_id from rank");

    assert(recv(conns[rank], &len, sizeof(uint64_t), MSG_WAITALL) != -1);
    assert(len == len0 && "inconsistent len from rank");
  }

  // recv data
  void *data = malloc(len * nranks);
  for (uint32_t rank = 0; rank < nranks; rank++) {
    char *ptr = (char *)data + rank * len;
    assert(recv(conns[rank], ptr, len, MSG_WAITALL) != -1);
  }

  // send data to clients
  for (uint32_t rank = 0; rank < nranks; rank++) {
    assert(send(conns[rank], data, len * nranks, 0) != -1);
  }

  free(data);
}

void server_thread(int listenfd, uint32_t nranks) {
  int conns[nranks];

  for (uint32_t i = 0; i < nranks; i++) {
    // establish connection
    int conn = accept(listenfd, (struct sockaddr *)NULL, NULL);
    assert(conn != -1);

    // recv rank and save into conns
    uint32_t rank;
    assert(recv(conn, &rank, sizeof(uint32_t), MSG_WAITALL) != -1);
    conns[rank] = conn;
  }

  // send ack to clients
  uint32_t ack = 0;
  for (uint32_t i = 0; i < nranks; i++) {
    assert(send(conns[i], &ack, sizeof(uint32_t), 0) != -1);
  }

  while (true) {
    // receive a request from rank 0
    uint32_t request_id;
    auto ret = recv(conns[0], &request_id, sizeof(uint32_t), MSG_WAITALL);
    // recv 0 btyes means socket close
    if (ret == 0)
      break;
    assert(ret != -1 && "socket recv msg error");

    if (request_id == 1) {
      serve_barrier(nranks, conns);
    } else if (request_id == 2) {
      serve_broadcast(nranks, conns);
    } else if (request_id == 3) {
      serve_allgather(nranks, conns);
    } else {
      BRT_LOGS_DEFAULT(ERROR) << "unexpected request id:" << request_id;
      BRT_THROW("unexpected error");
    }
  }
}

} // namespace

Status CreateServer(uint32_t nranks, int port) {
  // create socket
  int listenfd = socket(AF_INET, SOCK_STREAM, 0);
  assert(listenfd != -1);

  // set server_addr
  struct sockaddr_in server_addr;
  memset(&server_addr, 0, sizeof(server_addr));
  server_addr.sin_family = AF_INET;
  server_addr.sin_addr.s_addr = htonl(INADDR_ANY);
  server_addr.sin_port = htons(port);

  // bind and listen
  int opt = 1;
  assert(setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(int)) !=
         -1);
  assert(bind(listenfd, (struct sockaddr *)&server_addr, sizeof(server_addr)) !=
         -1);
  assert(listen(listenfd, nranks) != -1);

  // start server thread
  std::thread th(server_thread, listenfd, nranks);
  th.detach();

  return Status::OK();
}

//===----------------------------------------------------------------------===//
// RendezvousSocket
//===----------------------------------------------------------------------===//

RendezvousSocket::RendezvousSocket(uint32_t nranks, uint32_t rank)
    : nranks_(nranks), rank_(rank), connected_(false) {}

RendezvousSocket::~RendezvousSocket() {}

Status RendezvousSocket::connect(const char *master_ip, int port) {
  std::unique_lock<std::mutex> lock(mutex_);

  if (connected_) {
    return Status(common::StatusCategory::BRT, common::StatusCode::FAIL,
                  "Client already connected");
  }

  // create socket
  conn_ = socket(AF_INET, SOCK_STREAM, 0);
  assert(conn_ != -1);

  // set server_addr
  struct sockaddr_in server_addr;
  memset(&server_addr, 0, sizeof(server_addr));
  server_addr.sin_family = AF_INET;
  server_addr.sin_port = htons(port);
  assert(inet_pton(AF_INET, master_ip, &server_addr.sin_addr) != -1);

  // connect
  int ret =
      ::connect(conn_, (struct sockaddr *)&server_addr, sizeof(server_addr));
  while (ret == -1) {
    usleep(100000); // 100ms
    ret =
        ::connect(conn_, (struct sockaddr *)&server_addr, sizeof(server_addr));
  }

  // send client rank
  assert(send(conn_, &rank_, sizeof(uint32_t), 0) != -1);

  // recv ack from server
  uint32_t ack;
  assert(recv(conn_, &ack, sizeof(uint32_t), MSG_WAITALL) != -1);

  connected_ = true;
  return Status::OK();
}

Status RendezvousSocket::barrier() {
  std::unique_lock<std::mutex> lock(mutex_);

  if (!connected_) {
    return Status(common::StatusCategory::BRT, common::StatusCode::FAIL,
                  "Client not connected");
  }

  // send request_id
  uint32_t request_id = 1;
  assert(send(conn_, &request_id, sizeof(uint32_t), 0) != -1);

  // recv ack
  uint32_t ack;
  assert(recv(conn_, &ack, sizeof(uint32_t), MSG_WAITALL) != -1);

  return Status::OK();
}

Status RendezvousSocket::broadcast(const void *sendbuff, void *recvbuff,
                                   size_t len, uint32_t root) {
  std::unique_lock<std::mutex> lock(mutex_);

  if (!connected_) {
    return Status(common::StatusCategory::BRT, common::StatusCode::FAIL,
                  "Client not connected");
  }

  // send request_id
  uint32_t request_id = 2;
  assert(send(conn_, &request_id, sizeof(uint32_t), 0) != -1);

  // send root
  assert(send(conn_, &root, sizeof(uint32_t), 0) != -1);

  // send len
  uint64_t len64 = len;
  assert(send(conn_, &len64, sizeof(uint64_t), 0) != -1);

  // send data
  if (rank_ == root) {
    assert(send(conn_, sendbuff, len, 0) != -1);
  }

  // recv data
  assert(recv(conn_, recvbuff, len, MSG_WAITALL) != -1);

  return Status::OK();
}

Status RendezvousSocket::allgather(const void *sendbuff, void *recvbuff,
                                   size_t sendlen) {
  std::unique_lock<std::mutex> lock(mutex_);

  if (!connected_) {
    return Status(common::StatusCategory::BRT, common::StatusCode::FAIL,
                  "Client not connected");
  }

  // send request_id
  uint32_t request_id = 3;
  assert(send(conn_, &request_id, sizeof(uint32_t), 0) != -1);

  // send sendlen
  uint64_t sendlen64 = sendlen;
  assert(send(conn_, &sendlen64, sizeof(uint64_t), 0) != -1);

  // send data
  assert(send(conn_, sendbuff, sendlen, 0) != -1);

  // recv data
  assert(recv(conn_, recvbuff, sendlen * nranks_, MSG_WAITALL) != -1);

  return Status::OK();
}

} // namespace brt