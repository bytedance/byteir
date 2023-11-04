// Copyright (c) Megvii Inc.
// Licensed under Apache License, Version 2.0
// ===========================================================================
// Modification Copyright 2022 ByteDance Ltd. and/or its affiliates.

#include "brt/backends/cuda/device/distributed/distributed_backend_nccl.h"
#include "brt/backends/cuda/device/distributed/d_context_nccl.h"
#include "brt/backends/cuda/device/distributed/utils.h"
#include "brt/core/common/logging/logging.h"

#include "nccl.h"
#include <cassert>

#define CHECK_LAUNCH_MODE                                                      \
  do {                                                                         \
    const char *str = getenv("NCCL_LAUNCH_MODE");                              \
    if (!str or strcmp(str, "PARALLEL") != 0) {                                \
      BRT_LOGS_DEFAULT(ERROR)                                                  \
          << "please set NCCL_LAUNCH_MODE to \"PARALLEL\"\n";                  \
      BRT_THROW("nccl error");                                                 \
    }                                                                          \
  } while (0)

namespace brt {

class DistributedBackendNCCLPrivate {
public:
  ncclComm_t m_comm;
  ~DistributedBackendNCCLPrivate() { ncclCommDestroy(m_comm); }
};

DistributedBackendNCCL::DistributedBackendNCCL(int nranks, int rank)
    : DistributedBackend(nranks, rank) {}

DistributedBackendNCCL::~DistributedBackendNCCL() {}

Status DistributedBackendNCCL::do_init() {
  uint32_t root = 0;
  ncclUniqueId uid;
  if (m_rank == root) {
    ncclGetUniqueId(&uid);
  }
  auto status = m_client->broadcast(&uid, &uid, NCCL_UNIQUE_ID_BYTES, root);
  if (status != Status::OK())
    return status;
  m_nccl = std::make_unique<DistributedBackendNCCLPrivate>();
  NCCL_ASSERT(ncclCommInitRank(&m_nccl->m_comm, m_nranks, uid, m_rank));
  return Status::OK();
}

Status DistributedBackendNCCL::do_init(BcastCallback cb) {
  uint32_t root = 0;
  ncclUniqueId uid;
  if (m_rank == root) {
    ncclGetUniqueId(&uid);
  }
  cb(uid.internal, NCCL_UNIQUE_ID_BYTES);
  m_nccl = std::make_unique<DistributedBackendNCCLPrivate>();
  NCCL_ASSERT(ncclCommInitRank(&m_nccl->m_comm, m_nranks, uid, m_rank));
  return Status::OK();
}

Status DistributedBackendNCCL::_send(const void *sendbuff, size_t size,
                                     uint32_t rank,
                                     std::shared_ptr<DContext> ctx) {
  // check context type and get cuda stream
  assert(ctx->type() == "BRT_CTX_CUDA" && "only cuda context supported");
  cudaStream_t stream = static_cast<CudaContext *>(ctx.get())->get_stream();
  // perform nccl send synchronously
  NCCL_ASSERT(ncclSend(sendbuff, size, ncclChar, rank, m_nccl->m_comm, stream));
  return Status::OK();
}

Status DistributedBackendNCCL::_recv(void *recvbuff, size_t size, uint32_t rank,
                                     std::shared_ptr<DContext> ctx) {
  // check context type and get cuda stream
  assert(ctx->type() == "BRT_CTX_CUDA" && "only cuda context supported");
  cudaStream_t stream = static_cast<CudaContext *>(ctx.get())->get_stream();
  // perform nccl send synchronously
  NCCL_ASSERT(ncclRecv(recvbuff, size, ncclChar, rank, m_nccl->m_comm, stream));
  return Status::OK();
}

Status DistributedBackendNCCL::scatter(const void *sendbuff, void *recvbuff,
                                       size_t recvlen, DType dtype,
                                       uint32_t root,
                                       std::shared_ptr<DContext> ctx) {
  // check context type and get cuda stream
  assert(ctx->type() == "BRT_CTX_CUDA" && "only cuda context supported");
  cudaStream_t stream = static_cast<CudaContext *>(ctx.get())->get_stream();
  ncclDataType_t nccl_dtype = get_nccl_dtype(dtype);
  CHECK_LAUNCH_MODE;
  // perform nccl send/recv in a group
  ncclGroupStart();
  if (m_rank == root) {
    for (size_t r = 0; r < m_nranks; r++) {
      const char *p =
          (const char *)sendbuff + r * recvlen * GetDTypeSize(dtype);
      NCCL_ASSERT(ncclSend((const void *)p, recvlen, nccl_dtype, r,
                           m_nccl->m_comm, stream));
    }
  }
  NCCL_ASSERT(
      ncclRecv(recvbuff, recvlen, nccl_dtype, root, m_nccl->m_comm, stream));
  ncclGroupEnd();
  return Status::OK();
}

Status DistributedBackendNCCL::gather(const void *sendbuff, void *recvbuff,
                                      size_t sendlen, DType dtype,
                                      uint32_t root,
                                      std::shared_ptr<DContext> ctx) {
  // check context type and get cuda stream
  assert(ctx->type() == "BRT_CTX_CUDA" && "only cuda context supported");
  cudaStream_t stream = static_cast<CudaContext *>(ctx.get())->get_stream();
  ncclDataType_t nccl_dtype = get_nccl_dtype(dtype);
  CHECK_LAUNCH_MODE;
  // perform nccl send/recv in a group
  ncclGroupStart();
  if (m_rank == root) {
    for (size_t r = 0; r < m_nranks; r++) {
      char *p = (char *)recvbuff + r * sendlen * GetDTypeSize(dtype);
      NCCL_ASSERT(
          ncclRecv((void *)p, sendlen, nccl_dtype, r, m_nccl->m_comm, stream));
    }
  }
  NCCL_ASSERT(
      ncclSend(sendbuff, sendlen, nccl_dtype, root, m_nccl->m_comm, stream));
  ncclGroupEnd();
  return Status::OK();
}

Status DistributedBackendNCCL::all_to_all(const void *sendbuff, void *recvbuff,
                                          size_t len, DType dtype,
                                          std::shared_ptr<DContext> ctx) {
  // check context type and get cuda stream
  assert(ctx->type() == "BRT_CTX_CUDA" && "only cuda context supported");
  cudaStream_t stream = static_cast<CudaContext *>(ctx.get())->get_stream();
  ncclDataType_t nccl_dtype = get_nccl_dtype(dtype);
  CHECK_LAUNCH_MODE;
  // perform nccl send/recv in a group
  ncclGroupStart();
  for (size_t r = 0; r < m_nranks; r++) {
    const char *p = (const char *)sendbuff + r * len * GetDTypeSize(dtype);
    char *q = (char *)recvbuff + r * len * GetDTypeSize(dtype);
    NCCL_ASSERT(
        ncclSend((const void *)p, len, nccl_dtype, r, m_nccl->m_comm, stream));
    NCCL_ASSERT(
        ncclRecv((void *)q, len, nccl_dtype, r, m_nccl->m_comm, stream));
  }
  ncclGroupEnd();
  return Status::OK();
}

Status DistributedBackendNCCL::all_gather(const void *sendbuff, void *recvbuff,
                                          size_t sendlen, DType dtype,
                                          std::shared_ptr<DContext> ctx) {
  // check context type and get cuda stream
  assert(ctx->type() == "BRT_CTX_CUDA" && "only cuda context supported");
  cudaStream_t stream = static_cast<CudaContext *>(ctx.get())->get_stream();
  // perform all gather synchronously
  NCCL_ASSERT(ncclAllGather(sendbuff, recvbuff, sendlen, get_nccl_dtype(dtype),
                            m_nccl->m_comm, stream));
  return Status::OK();
}

Status DistributedBackendNCCL::all_reduce(const void *sendbuff, void *recvbuff,
                                          size_t len, DType dtype, ReduceOp op,
                                          std::shared_ptr<DContext> ctx) {
  // check context type and get cuda stream
  assert(ctx->type() == "BRT_CTX_CUDA" && "only cuda context supported");
  cudaStream_t stream = static_cast<CudaContext *>(ctx.get())->get_stream();
  // perform all reduce synchronously
  NCCL_ASSERT(ncclAllReduce(sendbuff, recvbuff, len, get_nccl_dtype(dtype),
                            get_nccl_reduce_op(op), m_nccl->m_comm, stream));
  return Status::OK();
}

Status DistributedBackendNCCL::reduce_scatter(const void *sendbuff,
                                              void *recvbuff, size_t recvlen,
                                              DType dtype, ReduceOp op,
                                              std::shared_ptr<DContext> ctx) {
  // check context type and get cuda stream
  assert(ctx->type() == "BRT_CTX_CUDA" && "only cuda context supported");
  cudaStream_t stream = static_cast<CudaContext *>(ctx.get())->get_stream();
  // perform reduce scatter synchronously
  NCCL_ASSERT(ncclReduceScatter(sendbuff, recvbuff, recvlen,
                                get_nccl_dtype(dtype), get_nccl_reduce_op(op),
                                m_nccl->m_comm, stream));
  return Status::OK();
}

Status DistributedBackendNCCL::broadcast(const void *sendbuff, void *recvbuff,
                                         size_t len, DType dtype, uint32_t root,
                                         std::shared_ptr<DContext> ctx) {
  // check context type and get cuda stream
  assert(ctx->type() == "BRT_CTX_CUDA" && "only cuda context supported");
  cudaStream_t stream = static_cast<CudaContext *>(ctx.get())->get_stream();
  // perform broadcast synchronously
  NCCL_ASSERT(ncclBroadcast(sendbuff, recvbuff, len, get_nccl_dtype(dtype),
                            root, m_nccl->m_comm, stream));
  return Status::OK();
}

Status DistributedBackendNCCL::reduce(const void *sendbuff, void *recvbuff,
                                      size_t len, DType dtype, ReduceOp op,
                                      uint32_t root,
                                      std::shared_ptr<DContext> ctx) {
  // check context type and get cuda stream
  assert(ctx->type() == "BRT_CTX_CUDA" && "only cuda context supported");
  cudaStream_t stream = static_cast<CudaContext *>(ctx.get())->get_stream();
  // perform reduce synchronously
  NCCL_ASSERT(ncclReduce(sendbuff, recvbuff, len, get_nccl_dtype(dtype),
                         get_nccl_reduce_op(op), root, m_nccl->m_comm, stream));
  return Status::OK();
}

Status DistributedBackendNCCL::group_start() {
  CHECK_LAUNCH_MODE;
  ncclGroupStart();
  return Status::OK();
}

Status DistributedBackendNCCL::group_end() {
  CHECK_LAUNCH_MODE;
  ncclGroupEnd();
  return Status::OK();
}

} // namespace brt