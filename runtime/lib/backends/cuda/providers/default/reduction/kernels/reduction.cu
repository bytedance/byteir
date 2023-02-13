// Copyright (c) Megvii Inc.
// Licensed under the Apache License.
// ===========================================================================
// Modification Copyright 2022 ByteDance Ltd. and/or its affiliates.

#include "./reduction.h"
#include "./reduction_helper.h"
#include <algorithm>
#include <cstdint>

#define DIVUP(x, y) (((x) + (y)-1) / (y))

namespace brt {
namespace cuda {
namespace kernel {
namespace reduction {
/*!
 * each block has (1 << block_size_log2) threads and process fixed number of
 * rows; each row is processed by (1 << nr_thread_per_row_log2) threads.
 *
 * need a padding of max_nr_threads_per_row/2 elements after shared memory
 */
template <int block_size_log2, int max_nr_threads_per_row, class Op,
          int warp_size>
__global__ void kern_column(Op op, uint32_t A, uint32_t B,
                            uint32_t nr_thread_per_row_log2,
                            uint32_t sm_width_byte) {
  typedef typename Op::wtype wtype;
  // shared mem: matrix(nr_row_per_block, nr_thread_per_row)
  extern __shared__ uint8_t sub_block_raw[];

  uint32_t nr_row_per_block = 1 << (block_size_log2 - nr_thread_per_row_log2),
           nr_thread_per_row = 1 << nr_thread_per_row_log2,
           row_num = threadIdx.x >> nr_thread_per_row_log2,
           // tid in current row
      tid = threadIdx.x - (row_num << nr_thread_per_row_log2),
           a = blockIdx.x * nr_row_per_block + row_num;

  volatile wtype *row = (wtype *)(sub_block_raw + row_num * sm_width_byte);
  // sum columns of src[a0:a1] and store in row
  {
    uint32_t base = min(a, A - 1) * B;
    wtype csum = op.read(base + tid);
    for (int c = tid + nr_thread_per_row; c < B; c += nr_thread_per_row) {
      csum = Op::apply(csum, op.read(base + c));
    }
    row[tid] = csum;
  }

#pragma unroll
  for (uint32_t i = max_nr_threads_per_row / 2; i; i >>= 1) {
    bool cond = nr_thread_per_row >= i * 2 && tid < i;
    if (i >= warp_size) {
      __syncthreads();
    } else {
      /**
       * \warning Since CUDA 9.0, for Volta and Turing architecture,
       * applications that assume reads and writes are implicitly visible
       * to other threads in same warp need to insert the new __syncwarp()
       * warp-wide barrier synchronization instruction between steps where
       * data is exchanged between threads via global or shared memory.
       * For details, please refer to
       * https://docs.nvidia.com/cuda/volta-tuning-guide/index.html
       */
      __syncwarp(0xffffffff);
    }
    if (cond) {
      wtype v0 = row[tid];
      wtype v1 = Op::apply(v0, row[tid + i]);
      row[tid] = v1;
    }
  }

  if (a < A && !tid) {
    op.write(a, row[0]);
  }
}

template <class Op, uint32_t max_nr_threads_per_row, uint32_t block_size_log2,
          uint32_t warp_size>
void _do_run_column(uint32_t A, uint32_t B, cudaStream_t stream, const Op &op) {
  typedef typename Op::wtype wtype;
  const uint32_t block_size = 1 << block_size_log2;
  uint32_t nr_thread_per_row = 1, nr_thread_per_row_log2 = 0;

  while (nr_thread_per_row < max_nr_threads_per_row &&
         nr_thread_per_row * 2 <= B) {
    ++nr_thread_per_row_log2;
    nr_thread_per_row *= 2;
  }
  // now: nr_thread_per_row <= B < nr_thread_per_row * 2

  if (B <= max_nr_threads_per_row * 4) {
    // find nr_thread_per_row with minimal wasted threads
    uint32_t min_cost = std::numeric_limits<uint32_t>::max(), min_cost_th = 0;
    for (uint32_t i = warp_size; i <= nr_thread_per_row; i *= 2) {
      uint32_t cost = (i - B % i) % i;
      if (cost < min_cost) {
        min_cost = cost;
        min_cost_th = i;
      }
    }
    if (min_cost_th) {
      nr_thread_per_row = min_cost_th;
      while ((1u << nr_thread_per_row_log2) != nr_thread_per_row)
        --nr_thread_per_row_log2;
    }
  }

  uint32_t nr_row_per_block = block_size / nr_thread_per_row,
           nr_blk = DIVUP(A, nr_row_per_block),
           sm_width_word32 = DIVUP(nr_thread_per_row * sizeof(wtype), 4ul);

  // gcd(sm_width_word32, BANKS) should be 1 to avoid bank confliction
  // iff sm_width_word32 is odd
  sm_width_word32 += !(sm_width_word32 % 2);
  uint32_t sm_width_byte = sm_width_word32 * 4,
           sm_size = nr_row_per_block * sm_width_byte +
                     sizeof(wtype) * max_nr_threads_per_row / 2;

  void (*kptr)(Op op, uint32_t A, uint32_t B, uint32_t nr_thread_per_row_log2,
               uint32_t sm_width_byte);
  if (nr_thread_per_row <= max_nr_threads_per_row / 4) {
    kptr =
        kern_column<block_size_log2, max_nr_threads_per_row / 4, Op, warp_size>;
  } else if (nr_thread_per_row <= max_nr_threads_per_row / 2) {
    kptr =
        kern_column<block_size_log2, max_nr_threads_per_row / 2, Op, warp_size>;
  } else {
    kptr = kern_column<block_size_log2, max_nr_threads_per_row, Op, warp_size>;
  }
  kptr<<<nr_blk, block_size, sm_size, stream>>>(
      op, A, B, nr_thread_per_row_log2, sm_width_byte);
}

// use struct to allow default template arguments in C++-03
/*!
 * \brief start the cuda kernel to reduce in column direction of a matrix
 * \tparam max_nr_threads_per_row max number of threads to reduce each row
 * \tparam block_size_log2 log2 of threads in a block
 * \tparam warp_size size of warp on the device
 */
template <class Op, uint32_t max_nr_threads_per_row = 64,
          uint32_t block_size_log2 = 7, uint32_t warp_size = 32>
struct run_column {
  static void run(uint32_t A, uint32_t B, cudaStream_t stream, const Op &op) {
    return _do_run_column<Op, max_nr_threads_per_row, block_size_log2,
                          warp_size>(A, B, stream, op);
  }
};

struct ExecPolicy {
  // (BY, BX) is the blockDim to launch reduce kernel
  ExecPolicy(size_t A, size_t B, size_t C) : A(A), B(B), C(C) {
    // use C to determine BX
    BX = 1;
    while (BX < 32 && BX < C)
      BX *= 2;
    BY = 512 / BX;
    NA = A;
    factor = BY * 4;
    NB = DIVUP(B, factor);
    NC = DIVUP(C, BX);
    {
      nr_reduces = 0;
      size_t tmp = B;
      while (tmp > 1) {
        tmp = DIVUP(tmp, factor);
        ++nr_reduces;
      }
      if (nr_reduces == 0)
        nr_reduces = 1;
    }
  }
  ExecPolicy next() const { return ExecPolicy(A, DIVUP(B, factor), C); }
  size_t factor;
  size_t nr_reduces;
  size_t BY, BX;
  size_t NA, NB, NC;
  size_t A, B, C;
};

// Whenever blockIdx is referenced, bidy_offset and bidz_offset should be added.
// This mechanism is to solve thread block size limitation issue by calling
// multiple kernels from host code.
template <class Operator, class Reader, class Writer, typename wtype,
          uint32_t BX, uint32_t BY, bool sync_within_warp>
__global__ void kern_largeBC(Operator opr, Reader rdr, Writer wtr, uint32_t A,
                             uint32_t B, uint32_t B2, uint32_t C,
                             uint32_t bidy_offset, uint32_t bidz_offset) {
  volatile __shared__ wtype shared[BY][BX];
  wtype s = opr.INIT;
  uint32_t c = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t a = blockIdx.z + bidz_offset;
  if (c < C) {
    uint32_t base = threadIdx.y + (blockIdx.y + bidy_offset) * 4 * blockDim.y;
    if (base + 0 * blockDim.y < B) {
      s = opr.apply(s, rdr.read(a * B * C + (base + 0 * blockDim.y) * C + c));
    }
    if (base + 1 * blockDim.y < B) {
      s = opr.apply(s, rdr.read(a * B * C + (base + 1 * blockDim.y) * C + c));
    }
    if (base + 2 * blockDim.y < B) {
      s = opr.apply(s, rdr.read(a * B * C + (base + 2 * blockDim.y) * C + c));
    }
    if (base + 3 * blockDim.y < B) {
      s = opr.apply(s, rdr.read(a * B * C + (base + 3 * blockDim.y) * C + c));
    }
  }
  shared[threadIdx.y][threadIdx.x] = s;
  __syncthreads();

  const uint32_t warp_y = 32 / BX;
#pragma unroll
  for (uint32_t k = 256; k > warp_y; k >>= 1) {
    if (BY >= k << 1) {
      if (threadIdx.y < k) {
        shared[threadIdx.y][threadIdx.x] =
            opr.apply(shared[threadIdx.y][threadIdx.x],
                      shared[threadIdx.y + k][threadIdx.x]);
      }
      __syncthreads();
    }
  }
  if (threadIdx.y < warp_y) {
#pragma unroll
    for (uint32_t k = warp_y; k > 0; k >>= 1) {
      if (threadIdx.y < k) {
        shared[threadIdx.y][threadIdx.x] =
            opr.apply(shared[threadIdx.y][threadIdx.x],
                      shared[threadIdx.y + k][threadIdx.x]);
      }
      if (sync_within_warp) {
        __syncthreads();
      }
      /**
       * \warning Since CUDA 9.0, for Volta and Turing architecture,
       * applications that assume reads and writes are implicitly visible
       * to other threads in same warp need to insert the new __syncwarp()
       * warp-wide barrier synchronization instruction between steps where
       * data is exchanged between threads via global or shared memory.
       * For details, please refer to
       * https://docs.nvidia.com/cuda/volta-tuning-guide/index.html
       */
      __syncwarp(0xffffffff);
    }
  }
  if (threadIdx.y == 0 && c < C) {
    uint32_t b2 = blockIdx.y + bidy_offset;
    wtr.write(a * B2 * C + b2 * C + c, shared[0][threadIdx.x]);
  }
}

/**
 * \tparam Operator must have method wtype apply(wtype, wtype)
 * \tparam Operator must have const member INIT
 * \tparam Reader must have method wtype read(size_t idx)
 * \tparam Writer must have method void write(size_t idx, wtype)
 */
template <class Operator, class Reader, class Writer, typename wtype,
          bool sync_within_warp>
void invoke_kernel(const ExecPolicy &p, const Operator &opr, const Reader &rdr,
                   const Writer &wtr, cudaStream_t stream) {
  // 32768 thread blocks for each call
#define CHECK(nBX)                                                             \
  if (p.BX == nBX && p.BY == 512 / nBX) {                                      \
    for (size_t bidy_offset = 0; bidy_offset < p.NB; bidy_offset += 32768)     \
      for (size_t bidz_offset = 0; bidz_offset < p.NA; bidz_offset += 32768) { \
        dim3 blocks;                                                           \
        blocks.x = p.NC;                                                       \
        blocks.y = std::min<size_t>(32768, p.NB - bidy_offset);                \
        blocks.z = std::min<size_t>(32768, p.NA - bidz_offset);                \
        kern_largeBC<Operator, Reader, Writer, wtype, nBX, 512 / nBX,          \
                     sync_within_warp>                                         \
            <<<blocks, dim3(p.BX, p.BY), 0, stream>>>(                         \
                opr, rdr, wtr, p.A, p.B, DIVUP(p.B, p.factor), p.C,            \
                bidy_offset, bidz_offset);                                     \
      }                                                                        \
  }
  CHECK(1);
  CHECK(2);
  CHECK(4);
  CHECK(8);
  CHECK(16);
  CHECK(32);
#undef CHECK
}

/**
 * inherit from PublicOperator
 */
template <class PublicOperator> struct PublicReader {
  PublicOperator opr;
  typedef typename PublicOperator::wtype wtype;
  PublicReader(const PublicOperator &opr) : opr(opr) {}
  __device__ wtype read(uint32_t idx) { return opr.read(idx); }
};

/**
 * read from workspace
 */
template <typename wtype> struct WorkspaceReader {
  wtype *workspace;
  WorkspaceReader(wtype *workspace) : workspace(workspace) {}
  __device__ wtype read(uint32_t idx) { return workspace[idx]; }
};

/**
 * inherit from PublicOperator
 */
template <class PublicOperator> struct PublicWriter {
  PublicOperator opr;
  typedef typename PublicOperator::wtype wtype;
  PublicWriter(const PublicOperator &opr) : opr(opr) {}
  __device__ void write(uint32_t idx, wtype value) { opr.write(idx, value); }
};

/**
 * write to workspace
 */
template <typename wtype> struct WorkspaceWriter {
  wtype *workspace;
  WorkspaceWriter(wtype *workspace) : workspace(workspace) {}
  __device__ void write(uint32_t idx, wtype value) { workspace[idx] = value; }
};

/**
 * \tparam PublicOperator
 *      must have typedef for wtype
 *      must have const static member wtype INIT
 *      must have method wtype read(uint32_t idx)
 *      must have method wtype apply(const wtype &, const wtype &)
 *      must have method void write(uint32_t idx, const wtype &)
 */
template <class PublicOperator, bool sync_within_warp>
void run_largeBC(typename PublicOperator::wtype *workspace, size_t A, size_t B,
                 size_t C, cudaStream_t stream, const PublicOperator &opr) {
  typedef typename PublicOperator::wtype wtype;
  ExecPolicy p(A, B, C);
  if (p.nr_reduces == 1) {
    PublicReader<PublicOperator> rdr(opr);
    PublicWriter<PublicOperator> wtr(opr);
    invoke_kernel<PublicOperator, PublicReader<PublicOperator>,
                  PublicWriter<PublicOperator>, wtype, sync_within_warp>(
        p, opr, rdr, wtr, stream);
  } else if (p.nr_reduces == 2) {
    PublicReader<PublicOperator> rdr1(opr);
    WorkspaceWriter<wtype> wtr1(workspace);
    WorkspaceReader<wtype> rdr2(workspace);
    PublicWriter<PublicOperator> wtr2(opr);
    invoke_kernel<PublicOperator, PublicReader<PublicOperator>,
                  WorkspaceWriter<wtype>, wtype, sync_within_warp>(
        p, opr, rdr1, wtr1, stream);
    p = p.next();
    invoke_kernel<PublicOperator, WorkspaceReader<wtype>,
                  PublicWriter<PublicOperator>, wtype, sync_within_warp>(
        p, opr, rdr2, wtr2, stream);
  } else {
    wtype *workspace1 = workspace;
    size_t B2 = DIVUP(B, p.factor);
    wtype *workspace2 = workspace + A * B2 * C;
    size_t nr_reduces = p.nr_reduces;

    {
      PublicReader<PublicOperator> rdr(opr);
      WorkspaceWriter<wtype> wtr(workspace1);
      invoke_kernel<PublicOperator, PublicReader<PublicOperator>,
                    WorkspaceWriter<wtype>, wtype, sync_within_warp>(
          p, opr, rdr, wtr, stream);
    }
    p = p.next();
    wtype *current = workspace1;
    wtype *next = workspace2;
    for (size_t i = 1; i < nr_reduces; ++i) {
      WorkspaceReader<wtype> rdr(current);
      if (i + 1 == nr_reduces) {
        PublicWriter<PublicOperator> wtr(opr);
        invoke_kernel<PublicOperator, WorkspaceReader<wtype>,
                      PublicWriter<PublicOperator>, wtype, sync_within_warp>(
            p, opr, rdr, wtr, stream);
      } else {
        WorkspaceWriter<wtype> wtr(next);
        invoke_kernel<PublicOperator, WorkspaceReader<wtype>,
                      WorkspaceWriter<wtype>, wtype, sync_within_warp>(
            p, opr, rdr, wtr, stream);
      }
      std::swap(next, current);
      p = p.next();
    }
  }
}

template <typename wtype>
size_t get_workspace_largeBC(size_t A, size_t B, size_t C) {
  ExecPolicy p(A, B, C);
  if (p.nr_reduces == 1) {
    // direct reduce
    return 0;
  } else if (p.nr_reduces == 2) {
    // src->workspace->dst
    size_t B2 = DIVUP(B, p.factor);
    return sizeof(wtype) * A * B2 * C;
  } else {
    // src->workspace1->workspace2->dst
    size_t B2 = DIVUP(B, p.factor);
    size_t B3 = DIVUP(B2, p.factor);
    return sizeof(wtype) * A * B2 * C + sizeof(wtype) * A * B3 * C;
  }
}

bool use_reduce_column(size_t A, size_t B, size_t C) {
  return C == 1 && (B <= A * 4 || B <= 32);
}
} // namespace reduction

template <typename T, typename Op>
void call_reduce(const T *input, T *output, size_t A, size_t B, size_t C,
                 void *workspace, cudaStream_t stream) {
  Op opr(const_cast<T *>(input), output, B);
  if (reduction::use_reduce_column(A, B, C)) {
    reduction::run_column<Op>::run(A, B, stream, opr);
  } else {
    reduction::run_largeBC<Op, false>(static_cast<float *>(workspace), A, B, C,
                                      stream, opr);
  }
}

template <typename wtype>
size_t get_reduce_workspace_in_bytes(size_t A, size_t B, size_t C) {
  if (reduction::use_reduce_column(A, B, C)) {
    return 0;
  }
  return reduction::get_workspace_largeBC<wtype>(A, B, C);
}

template void call_reduce<__half, reduction::SumOp<__half, __half, float>>(
    const __half *, __half *, size_t, size_t, size_t, void *, cudaStream_t);
template void call_reduce<float, reduction::SumOp<float, float, float>>(
    const float *, float *, size_t, size_t, size_t, void *, cudaStream_t);
template void call_reduce<float, reduction::MaxOp<float, float, float>>(
    const float *, float *, size_t, size_t, size_t, void *, cudaStream_t);
template void call_reduce<__half, reduction::MaxOp<__half, __half, float>>(
    const __half *, __half *, size_t, size_t, size_t, void *, cudaStream_t);
template size_t get_reduce_workspace_in_bytes<float>(size_t, size_t, size_t);
template size_t get_reduce_workspace_in_bytes<__half>(size_t, size_t, size_t);

} // namespace kernel
} // namespace cuda
} // namespace brt